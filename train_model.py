"""
AI Detector Model Trainer
Trains a classifier to distinguish between AI-generated and human-written text
"""

import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from training_data import get_training_data


class TextFeatureExtractor:
    """Extract custom features from text that distinguish AI from human writing"""
    
    @staticmethod
    def extract_features(text):
        """Extract statistical features from text"""
        features = {}
        
        # Basic statistics
        words = text.split()
        sentences = text.split('.')
        
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        features['avg_sentence_length'] = np.mean([len(sent.split()) for sent in sentences if sent.strip()]) if sentences else 0
        features['text_length'] = len(text)
        features['word_count'] = len(words)
        
        # Vocabulary richness
        unique_words = len(set(words))
        features['vocabulary_richness'] = unique_words / len(words) if words else 0
        
        # Punctuation usage
        features['comma_count'] = text.count(',')
        features['period_count'] = text.count('.')
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        
        # AI indicators (formal transitional phrases)
        ai_phrases = [
            'furthermore', 'moreover', 'consequently', 'therefore', 'additionally',
            'however', 'nevertheless', 'thus', 'hence', 'accordingly',
            # Chinese formal phrases
            '此外', '同時', '因此', '然而', '綜上所述', '隨著', '表明', '顯示'
        ]
        text_lower = text.lower()
        features['formal_transitions'] = sum(text_lower.count(phrase) for phrase in ai_phrases if phrase.isascii()) + sum(text.count(phrase) for phrase in ai_phrases if not phrase.isascii())
        
        # Human indicators (informal expressions)
        human_phrases = [
            'i think', 'i feel', 'honestly', 'actually', 'pretty much',
            'kind of', 'sort of', 'lol', 'idk', 'guess',
            # Chinese informal expressions
            '超級', '真的', '感覺', '哈哈', '欸', '啦', '吧', '喔', '什麼的'
        ]
        features['informal_expressions'] = sum(text_lower.count(phrase) for phrase in human_phrases if phrase.isascii()) + sum(text.count(phrase) for phrase in human_phrases if not phrase.isascii())
        
        return features


def train_model():
    """Train the AI detection model"""
    print("Loading training data...")
    texts, labels = get_training_data()
    
    print(f"Training on {len(texts)} samples...")
    print(f"  - AI-generated: {sum(labels)}")
    print(f"  - Human-written: {len(labels) - sum(labels)}")
    
    # Create pipeline with TF-IDF and Naive Bayes
    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=1500,  # Increased further
            ngram_range=(1, 4),  # Include up to 4-grams for better pattern matching
            min_df=1,
            max_df=0.95,  # More lenient
            token_pattern=r'(?u)\b\w+\b',  # Support both English and Chinese
            sublinear_tf=True,  # Use sublinear term frequency scaling
            analyzer='char_wb',  # Character-level analysis for better Chinese support
            lowercase=True
        )),
        ('classifier', MultinomialNB(alpha=0.3))  # Reduced alpha for more sensitivity
    ])
    
    # Cross-validation
    print("\nPerforming cross-validation...")
    scores = cross_val_score(model, texts, labels, cv=3, scoring='accuracy')
    print(f"Cross-validation accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
    
    # Train final model
    print("\nTraining final model...")
    model.fit(texts, labels)
    
    # Save model
    print("Saving model...")
    with open('ai_detector_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("\n✓ Model trained and saved successfully!")
    
    # Test predictions
    print("\n--- Sample Predictions ---")
    test_texts = [
        "Artificial intelligence has revolutionized the technology sector, fundamentally transforming operational paradigms.",
        "I just ate the best pizza ever! Seriously it was amazing and I can't wait to go back lol",
        "人工智慧技術在當代社會中扮演著重要角色。此外，深度學習算法已在多個領域取得顯著成就。因此，相關應用正在快速發展。",
        "最近在想AI這個東西真的很神奇欸。感覺以前很遙遠的技術現在突然就到處都是了。手機可以人臉辨識超酷的哈哈。"
    ]
    
    for text in test_texts:
        prediction = model.predict_proba([text])[0]
        print(f"\nText: {text[:60]}...")
        print(f"  AI: {prediction[1]*100:.1f}% | Human: {prediction[0]*100:.1f}%")
    
    return model


if __name__ == "__main__":
    train_model()
