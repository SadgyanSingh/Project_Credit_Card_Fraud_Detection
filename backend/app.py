from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, pymongo, bcrypt, jwt, datetime, re
from config import MONGO_URI, JWT_SECRET, JWT_ALGO

# Google Auth
from google.oauth2 import id_token
from google.auth.transport import requests as grequests

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# MongoDB setup
client = pymongo.MongoClient(MONGO_URI)
db = client["fraud_detection"]
users_collection = db["users"]

# Load trained ML model
model = joblib.load("credit_card_model.pkl")

# ---------------- AUTH UTILS ---------------- #
def create_token(email):
    payload = {"email": email, "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)

def verify_token(token):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
    except Exception:
        return None

# ---------------- ROUTES ---------------- #

@app.route("/")
def home():
    return "✅ Backend is live and working!"

# Signup
@app.route("/signup", methods=["POST"])
def signup():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"msg": "Email and password required"}), 400

    if users_collection.find_one({"email": email}):
        return jsonify({"msg": "User already exists"}), 400

    hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    users_collection.insert_one({"email": email, "password": hashed_pw})

    return jsonify({"msg": "Signup successful"}), 201


# Login
@app.route("/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"msg": "Email and password required"}), 400

    user = users_collection.find_one({"email": email})
    if user and bcrypt.checkpw(password.encode("utf-8"), user["password"].encode("utf-8")):
        token = create_token(email)
        return jsonify({"msg": "Login successful", "token": token})
    else:
        return jsonify({"msg": "Invalid credentials"}), 401


# Logout
@app.route("/logout", methods=["POST"])
def logout():
    # JWT is stateless, so backend just informs client to clear token
    return jsonify({"msg": "Logout successful, please clear token on client."}), 200


# Predict
@app.route("/predict", methods=["POST"])
def predict():
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return jsonify({"msg": "Missing or invalid token"}), 401

    token = auth_header.split(" ")[1]
    user_data = verify_token(token)

    if not user_data:
        return jsonify({"msg": "Token invalid or expired"}), 401

    features = request.json.get("features")
    prediction = model.predict([features])
    proba = model.predict_proba([features])[0][1]  # fraud probability
    return jsonify({
        "prediction": int(prediction[0]),
        "fraud_probability": float(proba)
    })


# Google Login
@app.route("/google-login", methods=["POST"])
def google_login():
    token_id = request.json.get("tokenId")
    try:
        idinfo = id_token.verify_oauth2_token(
            token_id,
            grequests.Request(),
            "105149798415-0b8s1bet2k0c2uce5ga4rifdmp04s572.apps.googleusercontent.com"
        )
        email = idinfo['email']
        if not users_collection.find_one({"email": email}):
            users_collection.insert_one({"email": email})
        token = create_token(email)
        return jsonify({"msg": "Google login successful", "token": token})
    except Exception as e:
        # Debug print
        print("Google verification error:", str(e))  
        return jsonify({"msg": "Google login failed", "error": str(e)}), 401


# ---------------- DYNAMIC CHATBOT ---------------- #

fraud_sample = [-2.3122265423263, 1.95199201064158, -1.60985073229769,
  3.9979055875468, -0.522187864667764, -1.42654531920595,
  -2.53738730624579, 1.39165724829804, -2.77008927719433,
  -2.77227214465915, 3.20203320709635, -2.89990738849473,
  -0.595221881324605, -4.28925378244217, 0.389724120274487,
  -1.14074717980657, -2.83005567450437, -0.0168224681808257,
  0.416955705037907, 0.126910559061474, 0.517232370861764,
  -0.0350493686052974, -0.465211076182388, 0.320198198514526,
  0.0445191674731724, 0.177839798284401, 0.261145002567677,
  -0.143275874698919, -0.35322939296682354]

normal_sample = [-1.3598071336738, -0.0727811733098497, 
  2.53634673796914, 1.37815522427443, -0.338320769942518, 
  0.462387777762292, 0.239598554061257, 0.0986979012610507, 
  0.363786969611213, 0.0907941719789316, -0.551599533260813,
  -0.617800855762348, -0.991389847235408, -0.311169353699879, 
  1.46817697209427, -0.470400525259478, 0.207971241929242, 
  0.0257905801985591, 0.403992960255733, 0.251412098239705, 
  -0.018306777944153, 0.277837575558899, -0.110473910188767, 
  0.0669280749146731, 0.128539358273528, -0.189114843888824, 
  0.133558376740387, -0.0210530534538215, 0.24496426337017338]


@app.route("/chatbot", methods=["POST"])
def chatbot():
    """
    Upgraded chatbot endpoint.
    - Better intent matching (regex + keywords)
    - Optional personalized greetings if valid JWT present in Authorization header
    - Returns structured JSON: {"reply": str, "type": "text|sample|prediction", "data": {...}}
    - Safely calls `model.predict` and `model.predict_proba` for sample predictions (uses existing model)
    """
    raw_msg = request.json.get("message", "")
    user_msg = (raw_msg or "").strip().lower()

    # Try to get user email from token (if provided) for personalization
    auth_header = request.headers.get("Authorization", "")
    user_email = None
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1]
        token_data = verify_token(token)
        if token_data:
            user_email = token_data.get("email")

    # helpers
    def text_response(text):
        return jsonify({"reply": text, "type": "text", "data": None})

    def sample_response(name, arr):
        return jsonify({
            "reply": f"Here is a {name} transaction sample (length {len(arr)}).",
            "type": "sample",
            "data": {"sample_name": name, "sample": arr}
        })

    def prediction_response(pred, proba):
        return jsonify({
            "reply": f"Prediction: {int(pred)} (fraud probability {proba:.4f})",
            "type": "prediction",
            "data": {"prediction": int(pred), "fraud_probability": float(proba)}
        })

    # default reply
    default = "Sorry, I didn’t understand 🤔 Try: 'hello', 'help', 'show fraud sample', 'show normal sample', 'predict fraud sample', 'dataset', or 'metrics'."

    # quick empty check
    if not user_msg:
        return text_response("Please send a message for the chatbot to respond to.")

    # greeting
    if re.search(r"\b(hi|hello|hey|yo|hiya)\b", user_msg):
        name_part = f" {user_email}" if user_email else ""
        return text_response(f"Hello{name_part} 👋 I'm Help AI. Ask me about the fraud detection app or type 'help' for commands.")

    # help
    if "help" in user_msg or "commands" in user_msg:
        help_text = (
            "I can help with:\n"
            "- 'show fraud sample' or 'fraud sample' -> returns an example fraud transaction.\n"
            "- 'show normal sample' or 'normal sample' -> returns an example normal transaction.\n"
            "- 'predict fraud sample' or 'predict normal sample' -> run model on the sample and return prediction + probability.\n"
            "- 'dataset' -> info about the dataset used.\n"
            "- 'metrics' or 'accuracy' -> model evaluation summary.\n"
            "- 'login' -> how to login.\n"
            "- 'contact' or 'support' -> support contact info."
        )
        return text_response(help_text)

    # dataset info
    if re.search(r"\bdataset\b", user_msg):
        return text_response("📊 We used the Kaggle credit card fraud dataset (284,807 transactions, 492 frauds).")

    # metrics / model info
    if re.search(r"\b(metrics|accuracy|precision|recall|f1|auc)\b", user_msg):
        return text_response("Our model (RandomForest + SMOTE) is evaluated using Precision, Recall, F1-score and AUC. Ask for 'metrics' to get more details from logs if available.")

    # login info
    if re.search(r"\b(login|sign in|signin|sign-in)\b", user_msg):
        return text_response("🔑 You can login using Email/Password at /login or use Google Sign-In at /google-login.")

    # contact/support
    if re.search(r"\b(contact|support|helpdesk|email)\b", user_msg):
        return text_response("📧 Reach us at support@frauddemo.com")

    # show samples: support variations like 'show fraud sample', 'fraud sample', 'get fraud sample 1' or 'show 1 fraud sample'
    if re.search(r"\bfraud\b.*\bsample\b", user_msg) or re.search(r"\bsample\b.*\bfraud\b", user_msg):
        return sample_response("fraud", fraud_sample)

    if re.search(r"\bnormal\b.*\bsample\b", user_msg) or re.search(r"\bsample\b.*\bnormal\b", user_msg):
        return sample_response("normal", normal_sample)

    # prediction on sample: "predict fraud sample" or "predict normal sample"
    if re.search(r"\bpredict\b.*\bfraud\b.*\bsample\b", user_msg) or re.search(r"\bpredict\b.*\bsample\b.*\bfraud\b", user_msg):
        try:
            if hasattr(model, "predict"):
                features = fraud_sample
                pred = model.predict([features])
                proba = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba([features])[0][1]
                else:
                    proba = float(pred[0])
                return prediction_response(pred[0], proba)
            else:
                return text_response("Model not available for predictions on the server.")
        except Exception as e:
            print("Prediction error (fraud sample):", e)
            return text_response(f"Prediction failed: {str(e)}")

    if re.search(r"\bpredict\b.*\bnormal\b.*\bsample\b", user_msg) or re.search(r"\bpredict\b.*\bsample\b.*\bnormal\b", user_msg):
        try:
            if hasattr(model, "predict"):
                features = normal_sample
                pred = model.predict([features])
                proba = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba([features])[0][1]
                else:
                    proba = float(pred[0])
                return prediction_response(pred[0], proba)
            else:
                return text_response("Model not available for predictions on the server.")
        except Exception as e:
            print("Prediction error (normal sample):", e)
            return text_response(f"Prediction failed: {str(e)}")

    # small fallback for 'predict' that points to /predict endpoint (safer)
    if "predict" in user_msg:
        return text_response("👉 To run predictions with your own features, call POST /predict with Authorization header and JSON {\"features\": [ ... ]}.")

    # final fallback
    return text_response(default)


if __name__ == "__main__":
    app.run(debug=True)
