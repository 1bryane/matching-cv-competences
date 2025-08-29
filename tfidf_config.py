#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚙️ Configuration TF-IDF pour le Matching de Compétences

Ce fichier contient les paramètres optimisés pour le vectoriseur TF-IDF
utilisé dans le système de matching de compétences.
"""

# Configuration du vectoriseur TF-IDF
TFIDF_CONFIG = {
    # Paramètres de base
    'stop_words': 'english',           # Mots communs à ignorer
    'max_features': 2000,              # Nombre maximum de features
    'ngram_range': (1, 3),            # Unigrammes, bigrammes et trigrammes
    'min_df': 1,                      # Terme doit apparaître au moins 1 fois
    'max_df': 0.95,                   # Terme ne doit pas apparaître dans plus de 95% des docs
    
    # Optimisations avancées
    'sublinear_tf': True,             # Application de log(1 + tf) pour réduire l'impact des termes très fréquents
    'analyzer': 'word',               # Analyse par mots
    'token_pattern': r'(?u)\b\w\w+\b', # Pattern pour les tokens (au moins 2 caractères)
    
    # Paramètres de performance
    'dtype': 'float32',               # Type de données pour optimiser la mémoire
    'norm': 'l2',                     # Normalisation L2 des vecteurs
}

# Configuration des seuils de qualité pour le matching
QUALITY_THRESHOLDS = {
    'high_quality': 0.6,              # Score minimum pour un match de haute qualité
    'medium_quality': 0.2,            # Score minimum pour un match de qualité moyenne
    'low_quality': 0.2,               # Score maximum pour un match de faible qualité
}

# Configuration du prétraitement des compétences
PREPROCESSING_CONFIG = {
    'min_token_length': 2,            # Longueur minimum des tokens
    'separators': [',', ';', '|', '/', '\n', '\r', '\t'],  # Séparateurs de compétences
    'remove_special_chars': True,     # Supprimer les caractères spéciaux
    'normalize_case': True,           # Normaliser la casse
}

# Configuration des n-grammes
NGRAM_CONFIG = {
    'min_n': 1,                       # N-gramme minimum
    'max_n': 3,                       # N-gramme maximum
    'weight_unigrams': 1.0,           # Poids des unigrammes
    'weight_bigrams': 1.2,            # Poids des bigrammes (plus important)
    'weight_trigrams': 1.5,           # Poids des trigrammes (très important)
}

# Configuration des métriques de similarité
SIMILARITY_CONFIG = {
    'metric': 'cosine',               # Métrique de similarité (cosine, euclidean, manhattan)
    'top_k_recommendations': 5,       # Nombre de recommandations top-k
    'similarity_threshold': 0.1,      # Seuil minimum de similarité pour considérer un match
}

# Configuration des stop words personnalisés pour le domaine IT
CUSTOM_STOP_WORDS = {
    'common_it_terms': [
        'and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'between', 'among'
    ],
    'technical_noise': [
        'version', 'release', 'update', 'patch', 'bug', 'fix', 'issue',
        'feature', 'tool', 'software', 'application', 'system', 'platform'
    ]
}

# Configuration des synonymes et acronymes courants
SYNONYMS_CONFIG = {
    'programming_languages': {
        'js': 'javascript',
        'py': 'python',
        'cpp': 'c++',
        'csharp': 'c#',
        'ts': 'typescript',
        'rb': 'ruby',
        'php': 'php',
        'go': 'golang',
        'rs': 'rust',
        'swift': 'swift'
    },
    'frameworks': {
        'react': 'reactjs',
        'angular': 'angularjs',
        'vue': 'vuejs',
        'express': 'expressjs',
        'django': 'djangoframework',
        'flask': 'flaskframework',
        'spring': 'springframework',
        'hibernate': 'hibernateorm'
    },
    'databases': {
        'postgres': 'postgresql',
        'mysql': 'mysqlserver',
        'mongo': 'mongodb',
        'redis': 'rediscache',
        'sqlite': 'sqlite3'
    }
}

# Configuration des métriques de performance
PERFORMANCE_CONFIG = {
    'cache_vectorizer': True,         # Mettre en cache le vectoriseur
    'batch_size': 100,                # Taille des lots pour le traitement
    'parallel_processing': True,      # Activer le traitement parallèle
    'memory_optimization': True,      # Optimiser l'utilisation mémoire
}

def get_tfidf_config():
    """Retourne la configuration TF-IDF complète"""
    return TFIDF_CONFIG.copy()

def get_quality_thresholds():
    """Retourne les seuils de qualité"""
    return QUALITY_THRESHOLDS.copy()

def get_preprocessing_config():
    """Retourne la configuration de prétraitement"""
    return PREPROCESSING_CONFIG.copy()

def get_synonyms_config():
    """Retourne la configuration des synonymes"""
    return SYNONYMS_CONFIG.copy()

def get_performance_config():
    """Retourne la configuration de performance"""
    return PERFORMANCE_CONFIG.copy()

if __name__ == "__main__":
    print("⚙️ Configuration TF-IDF chargée avec succès!")
    print(f"📊 Paramètres TF-IDF: {len(TFIDF_CONFIG)} paramètres")
    print(f"🎯 Seuils de qualité: {len(QUALITY_THRESHOLDS)} seuils")
    print(f"🔧 Prétraitement: {len(PREPROCESSING_CONFIG)} options")
    print(f"🔄 Synonymes: {len(SYNONYMS_CONFIG)} catégories")
    print(f"⚡ Performance: {len(PERFORMANCE_CONFIG)} optimisations")
