"""
Multi-Language NLP Text Classifier (Feature Engineering & SVM)
Author: Matheus Paixão
Domain: Natural Language Processing (NLP) & Pattern Recognition

Extracts 17 linguistic, orthographic, and morphological features from raw text
to classify sentences into English, Spanish, or Portuguese.
"""

import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Sample Corpora
ENGLISH_CORPUS = [
    "I am going to work right now.", "I love spending time with my family.",
    "Need to buy milk and bread.", "Let's go to the cinema on Saturday.",
    "I enjoy outdoor sports.", "Traffic is terrible today.", "The food was delicious!",
    "Have you ever visited New York?", "I have an important meeting tomorrow.",
    "The party starts at 8 PM.", "I am tired after a long day at work.",
    "Let's have a barbecue over the weekend.", "The book I am reading is very interesting.",
    "Learning to cook new dishes.", "I need to exercise regularly.",
    "Travelling abroad for my vacation.", "Do you like dancing?",
    "Today is my birthday!", "I enjoy listening to classical music.",
    "Studying for my final exams.", "My favorite sports team won the game.",
    "I want to learn playing guitar.", "Taking a road trip this summer.",
    "The park gets crowded on weekends.", "The movie I watched yesterday was great.",
    "Need to fix this issue as soon as possible.", "I love exploring new places.",
    "Visiting my grandparents on Sunday.", "Excited for summer vacation.",
    "Enjoy hiking in nature.", "The restaurant has an incredible view.",
    "Going out for dinner on Saturday."
]

SPANISH_CORPUS = [
    "Hola, ¿cómo estás?", "Me encanta leer libros.", "El clima está agradable hoy.",
    "¿Dónde está el restaurante más cercano?", "¿Qué hora es?", "Voy al parque todos los días.",
    "¿Puedes ajudarme con esto?", "Me gustaría ir de vacaciones.", "Este es mi libro favorito.",
    "Me gusta bailar salsa.", "¿Hablas español?", "¿Cuál es tu comida favorita?",
    "Estoy aprendiendo a tocar el piano.", "¡Que tengas un buen día!", "Necesito comprar algunas frutas.",
    "Vamos a dar un paseo.", "¿Cómo estuvo tu fin de semana?", "Estoy emocionado por el concierto.",
    "¿Me pasas la sal, por favor?", "Tengo una reunión a las 2 PM.", "Estoy planeando unas vacaciones.",
    "Ella canta hermosamente.", "El perro está jugando.", "Quiero aprender italiano.",
    "Disfruto ir a la playa.", "¿Dónde puedo encontrar un taxi?", "Lamento las molestias.",
    "Estoy estudiando para mis exámenes.", "Me gusta cocinar la cena en casa.",
    "¿Tienes alguna recomendación de restaurantes?"
]

PORTUGUESE_CORPUS = [
    "Estou indo para o trabalho agora.", "Adoro passar tempo com minha família.",
    "Preciso comprar leite e pão.", "Vamos ao cinema no sábado.",
    "Gosto de praticar esportes ao ar livre.", "O trânsito está terrível hoje.",
    "A comida estava deliciosa!", "Você já visitou o Rio de Janeiro?",
    "Tenho uma reunião importante amanhã.", "A festa começa às 20h.",
    "Estou cansado depois de um longo dia de trabalho.", "Vamos fazer um churrasco no final de semana.",
    "O livro que estou lendo é muito interessante.", "Estou aprendendo a cozinhar pratos novos.",
    "Preciso fazer exercícios físicos regularmente.", "Vou viajar para o exterior nas férias.",
    "Você gosta de dançar?", "Hoje é meu aniversário!",
    "Gosto de ouvir música clássica.", "Estou estudando para o vestibular.",
    "Meu time de futebol favorito ganhou o jogo.", "Quero aprender a tocar violão.",
    "Vamos fazer uma viagem de carro.", "O parque fica cheio aos finais de semana.",
    "O filme que assisti ontem foi ótimo.", "Preciso resolver esse problema o mais rápido possível.",
    "Adoro explorar novos lugares.", "Vou visitar meus avós no domingo.",
    "Estou ansioso para as férias de verão.", "Gosto de fazer caminhadas na natureza.",
    "O restaurante tem uma vista incrível.", "Vamos sair para jantar no sábado."
]


def extract_features(text: str) -> list:
    """Extracts 17 linguistic and orthographic features from a raw text string."""
    clean_text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    words = [w for w in clean_text.split() if len(w) > 0]
    avg_word_len = sum(len(w) for w in words) / len(words) if words else 0.0
    
    features = [
        avg_word_len,
        1.0 if re.search(r"[¿¡]", text) else 0.0,
        1.0 if 'ç' in text.lower() else 0.0,
        1.0 if 'ñ' in text.lower() else 0.0,
        1.0 if text.lower().endswith("ção") else 0.0,
        1.0 if text.lower().endswith("ción") else 0.0,
        1.0 if re.search(r"\b\w+'\w+\b", text) else 0.0,
        1.0 if re.search(r"\b(the|how|is|and|with|my|to|you)\b", text, re.IGNORECASE) else 0.0,
        1.0 if re.search(r"\b(el|la|que|de|y|hola|cómo|estás)\b", text, re.IGNORECASE) else 0.0,
        1.0 if re.search(r"\b(que|não|por|para|estou|com|uma)\b", text, re.IGNORECASE) else 0.0,
        float(sum(text.lower().count(c) for c in ['ã', 'õ', 'â', 'ê', 'ô', 'á', 'é', 'í', 'ó', 'ú', 'ç'])),
        float(sum(text.lower().count(c) for c in ['á', 'é', 'í', 'ó', 'ú', 'ñ'])),
        float(text.lower().count('ñ') + text.lower().count('ã') + text.lower().count('õ')),
        1.0 if re.search(r"\b(el|la|los|las|un|una|unos|unas)\b", text, re.IGNORECASE) else 0.0,
        1.0 if any(p in text.lower() for p in ['¿qué', '¿dónde', '¿cuál', '¿cómo']) else 0.0,
        1.0 if any(p in text.lower() for p in ['o que', 'onde', 'qual', 'como']) else 0.0,
        1.0 if re.search(r"\b(o|a|os|as|um|uma|uns|umas)\b", text, re.IGNORECASE) else 0.0,
    ]
    return features


def build_dataset():
    """Assembles corpus and extracts feature matrix X and target y."""
    records = []
    for s in ENGLISH_CORPUS:
        records.append((s, "English"))
    for s in SPANISH_CORPUS:
        records.append((s, "Spanish"))
    for s in PORTUGUESE_CORPUS:
        records.append((s, "Portuguese"))
        
    X_list = [extract_features(r[0]) for r in records]
    y_list = [r[1] for r in records]
    return np.array(X_list), np.array(y_list)


def train_language_classifier():
    """Trains SVM classifier and outputs evaluation metrics."""
    X, y = build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    
    model = SVC(kernel="linear", C=1.0)
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"[+] Model Training Completed.")
    print(f"[+] Train Accuracy: {train_acc * 100:.2f}% | Test Accuracy: {test_acc * 100:.2f}%")
    
    preds = model.predict(X_test)
    print("\n--- Classification Report ---")
    print(classification_report(y_test, preds))
    
    return model


def predict_sentence_language(model, text: str) -> str:
    """Predicts language for a single input text string."""
    feats = np.array([extract_features(text)])
    prediction = model.predict(feats)[0]
    return prediction


if __name__ == "__main__":
    trained_model = train_language_classifier()
    
    # Test sample inference
    test_samples = [
        "How are you doing today?",
        "Hola, ¿dónde está a biblioteca?",
        "Estou aprendendo ciência de dados e inteligência artificial."
    ]
    
    print("\n--- Sample Predictions ---")
    for sample in test_samples:
        lang = predict_sentence_language(trained_model, sample)
        print(f"Text: '{sample}' => Predicted Language: {lang}")
