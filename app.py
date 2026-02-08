
import gradio as gr
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import json
import os

# --- CONFIGURATION ---
# NOTE: When running locally, download models to a folder named 'models'
MODEL_PATH = "models" 

# --- LOAD MODELS ---
print("⏳ Loading AI Models...")
try:
    models = {
        'resnet50': keras.models.load_model(f"{MODEL_PATH}/resnet50_dogs_finetuned.keras"),
        'efficientnetb0': keras.models.load_model(f"{MODEL_PATH}/efficientnetb0_dogs_finetuned.keras"),
        'mobilenetv2': keras.models.load_model(f"{MODEL_PATH}/mobilenetv2_dogs_finetuned.keras")
    }
    mlp = keras.models.load_model(f"{MODEL_PATH}/mlp_ensemble.keras")
    
    with open(f"{MODEL_PATH}/class_indices.json", 'r') as f:
        class_indices = json.load(f)
    class_names = {v: k for k, v in class_indices.items()}
    print("✅ Models Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("Please ensure you have downloaded the models into a 'models' folder.")

# --- DATABASE (Traits + Life Span) ---
breed_db = {
    'Chihuahua': '14-16 years | Sassy, Charming, Graceful. Big personality in a tiny body.',
    'Japanese Spaniel': '10-12 years | Noble, Charming, Loving. An aristocratic lap dog.',
    'Maltese': '12-15 years | Gentle, Playful, Charming. Fearless toy dog with white hair.',
    'Pekinese': '12-14 years | Affectionate, Loyal, Regal. "Lion-like" independence.',
    'Shih Tzu': '10-18 years | Affectionate, Playful, Outgoing. A classic lap warmer.',
    'Blenheim Spaniel': '12-15 years | Affectionate, Gentle, Graceful. Also called Cavalier King Charles.',
    'Papillon': '14-16 years | Happy, Alert, Friendly. Famous for butterfly-like ears.',
    'Toy Terrier': '13-15 years | Spirited, Alert, Intelligent. A tiny watchdog.',
    'Rhodesian Ridgeback': '10-12 years | Dignified, Even-tempered, Affectionate. The African Lion Dog.',
    'Afghan Hound': '12-15 years | Independent, Sweet, Regal. Known for its silky coat.',
    'Basset': '12-13 years | Charming, Patient, Low-key. Famous for long ears and howling.',
    'Beagle': '10-15 years | Merry, Friendly, Curious. Follows its nose everywhere.',
    'Bloodhound': '10-12 years | Independent, Inquisitive, Friendly. The ultimate tracker.',
    'Bluetick': '11-12 years | Smart, Devoted, Tenacious. A nocturnal raccoon hunter.',
    'Black And Tan Coonhound': '10-12 years | Easy-going, Bright, Brave. Mellow at home, rugged on the trail.',
    'Walker Hound': '12-13 years | Smart, Brave, Courteous. A swift, tri-colored hunter.',
    'English Foxhound': '10-13 years | Gentle, Sociable, Athletic. Loves running in a pack.',
    'Redbone': '12-14 years | Even-tempered, Amiable, Eager. A laid-back red hunter.',
    'Borzoi': '9-14 years | Regal, Affectionate, Quiet. A "Greyhound with long hair".',
    'Irish Wolfhound': '6-8 years | Calm, Dignified, Courageous. A gentle giant.',
    'Italian Greyhound': '14-15 years | Alert, Playful, Sensitive. A miniature greyhound.',
    'Whippet': '12-15 years | Affectionate, Playful, Calm. Lightning fast but loves the couch.',
    'Ibizan Hound': '11-14 years | Family-oriented, Polite, Athletic. A deer-like jumper.',
    'Norwegian Elkhound': '12-15 years | Bold, Playful, Dependable. An ancient Viking dog.',
    'Otterhound': '10-13 years | Amiable, Boisterous, Even-tempered. A rare swimmer.',
    'Saluki': '10-17 years | Gentle, Dignified, Independent. The Royal Dog of Egypt.',
    'Scottish Deerhound': '8-11 years | Gentle, Dignified, Polite. The Royal Dog of Scotland.',
    'Weimaraner': '10-13 years | Friendly, Fearless, Obedient. The "Grey Ghost".',
    'Staffordshire Bullterrier': '12-14 years | Clever, Brave, Tenacious. Needs strong leadership.',
    'American Staffordshire Terrier': '12-16 years | Confident, Smart, Good-natured. A powerhouse companion.',
    'Bedlington Terrier': '11-16 years | Loyal, Charming, Frollicking. Looks like a lamb.',
    'Border Terrier': '12-15 years | Affectionate, Plucky, Happy. A tough little worker.',
    'Kerry Blue Terrier': '12-15 years | Alert, Adaptable, People-oriented. Non-shedding coat.',
    'Irish Terrier': '13-15 years | Bold, Dashing, Tenderhearted. The "Daredevil" of dogs.',
    'Norfolk Terrier': '12-16 years | Fearless, Alert, Gregarious. Drop-eared terrier.',
    'Norwich Terrier': '12-15 years | Affectionate, Alert, Curious. Prick-eared terrier.',
    'Yorkshire Terrier': '11-15 years | Sprightly, Tomboyish, Affectionate. Big dog in a small body.',
    'Wire Haired Fox Terrier': '12-15 years | Alert, Confident, Gregarious. A classic hunter.',
    'Lakeland Terrier': '12-16 years | Bold, Zesty, Friendly. A confident little dog.',
    'Sealyham Terrier': '12-14 years | Alert, Outgoing, Funny. The "couch potato" of terriers.',
    'Airedale': '11-14 years | Clever, Confident, Proud. The King of Terriers.',
    'Cairn': '13-15 years | Alert, Cheerful, Busy. Toto from The Wizard of Oz.',
    'Australian Terrier': '11-15 years | Courageous, Spirited, People-oriented. The first Aussie breed.',
    'Dandie Dinmont': '12-15 years | Independent, Smart, Determined. Unique "top-knot" hair.',
    'Boston Bull': '11-13 years | Friendly, Bright, Amusing. The "American Gentleman".',
    'Miniature Schnauzer': '12-15 years | Friendly, Smart, Obedient. A popular family dog.',
    'Giant Schnauzer': '12-15 years | Loyal, Alert, Trainable. A powerful worker.',
    'Standard Schnauzer': '13-16 years | Smart, Fearless, Spirited. The original Schnauzer.',
    'Scotch Terrier': '11-13 years | Confident, Independent, Spirited. The "Diehard".',
    'Tibetan Terrier': '15-16 years | Affectionate, Sensitive, Clever. Not actually a terrier!',
    'Silky Terrier': '13-15 years | Friendly, Quick, Alert. Silkier than a Yorkie.',
    'Soft Coated Wheaten Terrier': '12-14 years | Happy, Steady, Self-Confident. The Irish farm dog.',
    'West Highland White Terrier': '13-15 years | Happy, Loyal, Entertaining. The famous "Westie".',
    'Lhasa': '12-15 years | Confident, Smart, Comical. A Tibetan watchdog.',
    'Flat Coated Retriever': '8-10 years | Cheerful, Optimistic, Good-humored. The "Peter Pan" of dogs.',
    'Curly Coated Retriever': '10-12 years | Confident, Proud, Wickedly Smart. Water dog with curls.',
    'Golden Retriever': '10-12 years | Intelligent, Friendly, Devoted. The classic family dog.',
    'Labrador Retriever': '10-12 years | Active, Friendly, Outgoing. America\'s most popular breed.',
    'Chesapeake Bay Retriever': '10-13 years | Affectionate, Bright, Sensitive. A tough water dog.',
    'German Short Haired Pointer': '10-12 years | Friendly, Smart, Willing. An all-purpose hunter.',
    'Vizsla': '12-14 years | Affectionate, Gentle, Energetic. The "Velcro" dog.',
    'English Setter': '11-15 years | Friendly, Mellow, Merry. The "Gentleman of Dogs".',
    'Irish Setter': '12-15 years | Active, Outgoing, Sweet-natured. Flashy red coat.',
    'Gordon Setter': '12-13 years | Confident, Fearless, Alert. The heaviest setter.',
    'Brittany Spaniel': '12-14 years | Bright, Fun-loving, Upbeat. A versatile bird dog.',
    'Clumber': '10-12 years | Gentle, Loyal, Amusing. The largest spaniel.',
    'English Springer': '12-14 years | Friendly, Playful, Obedient. A classic flushing dog.',
    'Welsh Springer': '12-15 years | Active, Loyal, Affectionate. The red-and-white spaniel.',
    'Cocker Spaniel': '10-14 years | Gentle, Smart, Happy. The smallest sporting dog.',
    'Sussex Spaniel': '13-15 years | Calm, Steady, Affectionate. A low-slung hunter.',
    'Irish Water Spaniel': '12-13 years | Playful, Brave, Smart. The clown of the spaniel family.',
    'Kuvasz': '10-12 years | Protective, Loyal, Patient. A large white guardian.',
    'Schipperke': '13-15 years | Confident, Alert, Curious. The "Little Captain".',
    'Groenendael': '10-14 years | Intelligent, Protective, Intense. The black Belgian Shepherd.',
    'Malinois': '14-16 years | Confident, Smart, Hardworking. The ultimate police dog.',
    'Briard': '12 years | Confident, Smart, Faithful. A shaggy French herder.',
    'Kelpie': '10-13 years | Intelligent, Alert, Eager. An Australian workaholic.',
    'Komondor': '10-12 years | Steady, Fearless, Affectionate. The "Mop Dog".',
    'Old English Sheepdog': '10-12 years | Adaptable, Gentle, Smart. The shaggy dog.',
    'Shetland Sheepdog': '12-14 years | Playful, Energetic, Bright. A mini Lassie.',
    'Collie': '12-14 years | Devoted, Graceful, Proud. Famous for rescuing Timmy.',
    'Border Collie': '12-15 years | Remarkably Smart, Energetic. The world\'s smartest dog.',
    'Bouvier Des Flandres': '10-12 years | Strong-willed, Protective, Gentle. A rugged cattle herder.',
    'Rottweiler': '9-10 years | Loyal, Loving, Confident Guardian. Needs socialization.',
    'German Shepherd': '7-10 years | Confident, Courageous, Smart. A versatile worker.',
    'Doberman': '10-12 years | Alert, Fearless, Loyal. The "Tax Collector\'s Dog".',
    'Miniature Pinscher': '12-16 years | Fearless, Fun-loving, Proud. The "King of Toys".',
    'Greater Swiss Mountain Dog': '8-11 years | Faithful, Family-oriented. A Swiss draft dog.',
    'Bernese Mountain Dog': '7-10 years | Good-natured, Calm, Strong. A gentle giant.',
    'Appenzeller': '12-14 years | Reliable, Fearless, Lively. A Swiss cattle dog.',
    'Entlebucher': '11-13 years | Loyal, Smart, Enthusiastic. The smallest Swiss mountain dog.',
    'Boxer': '10-12 years | Bright, Fun-loving, Active. A muscular athlete.',
    'Bull Mastiff': '7-9 years | Affectionate, Courageous, Docile. A silent guardian.',
    'Tibetan Mastiff': '10-12 years | Independent, Reserved, Intelligent. A primitive guardian.',
    'French Bulldog': '10-12 years | Adaptable, Playful, Smart. Bat ears and a flat face.',
    'Great Dane': '7-10 years | Friendly, Patient, Dependable. The "Apollo of Dogs".',
    'Saint Bernard': '8-10 years | Playful, Charming, Inquisitive. The famous rescue dog.',
    'Eskimo Dog': '10-15 years | Alert, Friendly, Reserved. An Arctic worker.',
    'Malamute': '10-14 years | Affectionate, Loyal, Playful. A powerful sled dog.',
    'Siberian Husky': '12-14 years | Loyal, Outgoing, Mischievous. Born to run.',
    'Affenpinscher': '12-15 years | Confident, Famously Funny. The "Monkey Dog".',
    'Basenji': '13-14 years | Independent, Smart, Poised. The barkless dog.',
    'Pug': '13-15 years | Charming, Mischievous, Loving. A lot of dog in a small space.',
    'Leonberger': '7 years | Bright, Patient, Loving. A lion-like giant.',
    'Newfoundland': '9-10 years | Sweet, Patient, Devoted. A nanny dog.',
    'Great Pyrenees': '10-12 years | Smart, Patient, Calm. A majestic guardian.',
    'Samoyed': '12-14 years | Adaptable, Friendly, Gentle. The "Smiling Sammy".',
    'Pomeranian': '12-16 years | Inquisitive, Bold, Lively. A fluffy extrovert.',
    'Chow': '8-12 years | Dignified, Bright, Serious. Cat-like personality.',
    'Keeshond': '12-15 years | Friendly, Lively, Outgoing. The "Dutch Barge Dog".',
    'Brabancon Griffon': '12-15 years | Sensitive, Alert, Inquisitive. A bearded toy dog.',
    'Pembroke': '12-15 years | Affectionate, Smart, Alert. The Queen\'s favorite.',
    'Cardigan': '12-15 years | Affectionate, Loyal, Smart. The Corgi with a tail.',
    'Toy Poodle': '10-18 years | Agile, Intelligent, Self-Confident. A tiny athlete.',
    'Miniature Poodle': '10-18 years | Active, Proud, Very Smart. A mid-sized athlete.',
    'Standard Poodle': '10-18 years | Active, Proud, Very Smart. A large athlete.',
    'Mexican Hairless': '14-20 years | Loyal, Alert, Cheerful. The Xoloitzcuintli.',
    'Dingo': '15-20 years | Independent, Alert, Wild. The native dog of Australia.',
    'Dhole': '10-13 years | Social, Vocal, Wild. The Asiatic Wild Dog.',
    'African Hunting Dog': '10-12 years | Social, Intense, Wild. The Painted Wolf.'
}

# --- PREDICTION FUNCTION ---
def predict_dog_final(image):
    if image is None: return "Please upload an image.", ""

    # Preprocess
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img_arr = np.expand_dims(img, axis=0)

    # Predict
    p1 = models['resnet50'].predict(img_arr, verbose=0)
    p2 = models['efficientnetb0'].predict(img_arr, verbose=0)
    p3 = models['mobilenetv2'].predict(img_arr, verbose=0)
    
    # Ensemble Vote
    stacked_input = np.concatenate([p1, p2, p3], axis=1)
    final_probs = mlp.predict(stacked_input, verbose=0)[0]
    
    # Get Winner
    top_index = final_probs.argmax()
    confidence = final_probs[top_index] * 100
    
    # Cleaning Name
    raw_name = class_names[top_index]
    if '-' in raw_name:
        breed_name = raw_name.split('-', 1)[1]
    else:
        breed_name = raw_name
    breed_name = breed_name.replace('_', ' ').title()

    # Database Lookup
    info_found = "Details not found."
    lifespan = "Unknown"
    personality = "Unknown"
    search_key = breed_name.replace(" ", "").lower()
    
    for db_key, db_val in breed_db.items():
        if db_key.replace(" ", "").lower() in search_key or search_key in db_key.replace(" ", "").lower():
            breed_name = db_key
            parts = db_val.split('|')
            lifespan = parts[0].strip()
            personality = parts[1].strip()
            break

    # Uncertainty Check
    if confidence < 50.0:
        return {breed_name: confidence / 100}, f"⚠️ UNCERTAIN ({confidence:.1f}%)\n\nI'm not sure, but it looks a little like a {breed_name}."

    # Success Output
    info_text = (
        f"✅ MATCH FOUND!\n\n"
        f"🐕 Breed: {breed_name}\n"
        f"📊 Confidence: {confidence:.1f}%\n"
        f"⏳ Life Span: {lifespan}\n"
        f"📝 Personality: {personality}"
    )
    
    return {breed_name: confidence / 100}, info_text

# --- GRADIO INTERFACE ---
interface = gr.Interface(
    fn=predict_dog_final,
    inputs=gr.Image(sources=["upload", "webcam"], type="numpy", label="Upload or Snap"),
    outputs=[
        gr.Label(num_top_classes=3, label="Top Predictions"), 
        gr.Textbox(label="AI Analysis", lines=6)
    ],
    title="🐶 Ultimate Dog Classifier (120 Breeds)",
    description="Identifies 120 breeds with Lifespan & Personality.",
    theme="default"
)

if __name__ == "__main__":
    interface.launch()
