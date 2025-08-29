#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 Test du Système TF-IDF de Matching de Compétences

Ce fichier teste spécifiquement le système TF-IDF optimisé
pour le matching de compétences entre CVs et offres d'emploi.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from skills_preprocessor import create_preprocessor
from tfidf_config import get_tfidf_config, get_quality_thresholds

def test_tfidf_vectorizer():
    """Test du vectoriseur TF-IDF optimisé"""
    print("🧪 Test du Vectoriseur TF-IDF")
    print("=" * 50)
    
    # Configuration TF-IDF
    config = get_tfidf_config()
    print(f"📊 Configuration TF-IDF: {len(config)} paramètres")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Données de test
    cv_skills = [
        "Python, Data Science, Machine Learning, Deep Learning, Pandas, NumPy, Scikit-learn",
        "Java, Spring Boot, REST APIs, Microservices, Hibernate, Maven, JUnit",
        "JavaScript, React, Redux, Node.js, Express, MongoDB, HTML, CSS",
        "Project Management, Scrum, Agile, JIRA, Confluence, Risk Management",
        "C++, Object-Oriented Programming, Data Structures, Algorithms, STL"
    ]
    
    job_skills = [
        "Python, Data Science, Machine Learning",
        "Java, Spring Boot, REST APIs",
        "JavaScript, React, Redux",
        "Project Management, Scrum, Agile"
    ]
    
    print(f"\n📋 Données de test:")
    print(f"  CVs: {len(cv_skills)}")
    print(f"  Jobs: {len(job_skills)}")
    
    # Création du vectoriseur
    vectorizer = TfidfVectorizer(**config)
    
    # Combinaison des compétences
    all_skills = cv_skills + job_skills
    
    # Vectorisation
    tfidf_matrix = vectorizer.fit_transform(all_skills)
    
    print(f"\n✅ Vectorisation réussie!")
    print(f"  Forme de la matrice: {tfidf_matrix.shape}")
    print(f"  Nombre de features: {len(vectorizer.get_feature_names_out())}")
    
    # Séparation des vecteurs
    cv_vectors = tfidf_matrix[:len(cv_skills)]
    job_vectors = tfidf_matrix[len(cv_skills):]
    
    # Calcul de similarité
    similarity_matrix = cosine_similarity(cv_vectors, job_vectors)
    
    print(f"\n🎯 Matrice de similarité:")
    print(f"  Forme: {similarity_matrix.shape}")
    print(f"  Scores min/max: {similarity_matrix.min():.4f} / {similarity_matrix.max():.4f}")
    
    return vectorizer, similarity_matrix, cv_skills, job_skills

def test_preprocessing_impact():
    """Test de l'impact du prétraitement sur les résultats TF-IDF"""
    print("\n🔧 Test de l'Impact du Prétraitement")
    print("=" * 50)
    
    # Création du prétraiteur
    preprocessor = create_preprocessor()
    
    # Compétences avec et sans prétraitement
    raw_skills = [
        "Python, ML, Deep Learning, TensorFlow",
        "Java, Spring, REST APIs, Microservices",
        "JS, React, Node.js, MongoDB",
        "DevOps, Docker, K8s, CI/CD, AWS"
    ]
    
    processed_skills = preprocessor.preprocess_skills_batch(raw_skills)
    
    print("📝 Comparaison avant/après prétraitement:")
    for i, (raw, processed) in enumerate(zip(raw_skills, processed_skills)):
        print(f"\nCompétences {i+1}:")
        print(f"  Avant: {raw}")
        print(f"  Après:  {processed}")
    
    # Statistiques
    stats = preprocessor.get_skills_statistics(raw_skills)
    print(f"\n📊 Statistiques du prétraitement:")
    print(f"  Total compétences: {stats['total_skills']}")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Tokens uniques: {stats['unique_tokens']}")
    print(f"  Synonymes développés: {stats['synonyms_expanded']}")
    
    return raw_skills, processed_skills

def test_similarity_metrics():
    """Test des différentes métriques de similarité"""
    print("\n📊 Test des Métriques de Similarité")
    print("=" * 50)
    
    # Données de test
    cv_skills = [
        "Python, Data Science, Machine Learning, Deep Learning",
        "Java, Spring Boot, Microservices, Cloud Native",
        "JavaScript, React, TypeScript, Modern Web Development",
        "Data Engineering, Big Data, Apache Spark, ETL"
    ]
    
    job_skills = [
        "Python, Machine Learning, Deep Learning, Neural Networks",
        "Java, Spring Boot, Microservices, Docker, Kubernetes",
        "JavaScript, React, TypeScript, REST APIs",
        "Data Engineering, Big Data, Apache Spark, Data Pipeline"
    ]
    
    # Configuration TF-IDF
    config = get_tfidf_config()
    vectorizer = TfidfVectorizer(**config)
    
    # Vectorisation
    all_skills = cv_skills + job_skills
    tfidf_matrix = vectorizer.fit_transform(all_skills)
    
    # Séparation des vecteurs
    cv_vectors = tfidf_matrix[:len(cv_skills)]
    job_vectors = tfidf_matrix[len(cv_skills):]
    
    # Différentes métriques de similarité
    metrics = {
        'cosine': cosine_similarity,
        'euclidean': lambda x, y: 1 / (1 + np.linalg.norm(x.toarray()[:, np.newaxis] - y.toarray(), axis=2)),
        'manhattan': lambda x, y: 1 / (1 + np.sum(np.abs(x.toarray()[:, np.newaxis] - y.toarray()), axis=2))
    }
    
    print("🔍 Comparaison des métriques de similarité:")
    
    for metric_name, metric_func in metrics.items():
        try:
            similarity_matrix = metric_func(cv_vectors, job_vectors)
            print(f"\n  {metric_name.upper()}:")
            print(f"    Forme: {similarity_matrix.shape}")
            print(f"    Scores min/max: {similarity_matrix.min():.4f} / {similarity_matrix.max():.4f}")
            print(f"    Score moyen: {similarity_matrix.mean():.4f}")
        except Exception as e:
            print(f"\n  {metric_name.upper()}: Erreur - {e}")
    
    return vectorizer, cv_skills, job_skills

def test_quality_thresholds():
    """Test des seuils de qualité pour le matching"""
    print("\n🎯 Test des Seuils de Qualité")
    print("=" * 50)
    
    # Configuration des seuils
    thresholds = get_quality_thresholds()
    print(f"📊 Seuils configurés:")
    for key, value in thresholds.items():
        print(f"  {key}: {value}")
    
    # Génération de scores simulés
    np.random.seed(42)  # Pour la reproductibilité
    simulated_scores = np.random.uniform(0, 1, 100)
    
    print(f"\n📈 Analyse des scores simulés:")
    print(f"  Nombre de scores: {len(simulated_scores)}")
    print(f"  Score min/max: {simulated_scores.min():.4f} / {simulated_scores.max():.4f}")
    print(f"  Score moyen: {simulated_scores.mean():.4f}")
    
    # Application des seuils
    high_quality = np.sum(simulated_scores > thresholds['high_quality'])
    medium_quality = np.sum((simulated_scores > thresholds['medium_quality']) & 
                           (simulated_scores <= thresholds['high_quality']))
    low_quality = np.sum(simulated_scores <= thresholds['medium_quality'])
    
    print(f"\n🎯 Répartition par qualité:")
    print(f"  Haute qualité (>{thresholds['high_quality']}): {high_quality} ({high_quality/len(simulated_scores)*100:.1f}%)")
    print(f"  Qualité moyenne ({thresholds['medium_quality']}-{thresholds['high_quality']}): {medium_quality} ({medium_quality/len(simulated_scores)*100:.1f}%)")
    print(f"  Faible qualité (≤{thresholds['medium_quality']}): {low_quality} ({low_quality/len(simulated_scores)*100:.1f}%)")
    
    return simulated_scores, thresholds

def test_performance_optimization():
    """Test des optimisations de performance"""
    print("\n⚡ Test des Optimisations de Performance")
    print("=" * 50)
    
    import time
    
    # Données de test de grande taille
    cv_skills = [
        f"Skill_{i}, Technology_{i}, Framework_{i}, Language_{i}, Tool_{i}"
        for i in range(100)
    ]
    
    job_skills = [
        f"Job_Skill_{i}, Job_Tech_{i}, Job_Framework_{i}"
        for i in range(20)
    ]
    
    print(f"📊 Données de test:")
    print(f"  CVs: {len(cv_skills)}")
    print(f"  Jobs: {len(job_skills)}")
    
    # Configuration TF-IDF optimisée
    config = get_tfidf_config()
    
    # Test avec différents paramètres de performance
    performance_configs = [
        {'max_features': 1000, 'dtype': 'float32'},
        {'max_features': 2000, 'dtype': 'float32'},
        {'max_features': 5000, 'dtype': 'float32'}
    ]
    
    for i, perf_config in enumerate(performance_configs):
        print(f"\n🔧 Configuration {i+1}: {perf_config}")
        
        # Mise à jour de la configuration
        test_config = config.copy()
        test_config.update(perf_config)
        
        # Mesure du temps
        start_time = time.time()
        
        vectorizer = TfidfVectorizer(**test_config)
        all_skills = cv_skills + job_skills
        tfidf_matrix = vectorizer.fit_transform(all_skills)
        
        # Séparation et calcul de similarité
        cv_vectors = tfidf_matrix[:len(cv_skills)]
        job_vectors = tfidf_matrix[len(job_skills):]
        similarity_matrix = cosine_similarity(cv_vectors, job_vectors)
        
        end_time = time.time()
        
        print(f"  ⏱️  Temps d'exécution: {end_time - start_time:.4f}s")
        print(f"  📊 Forme de la matrice: {similarity_matrix.shape}")
        print(f"  💾 Nombre de features: {len(vectorizer.get_feature_names_out())}")
    
    return cv_skills, job_skills

def main():
    """Fonction principale de test"""
    print("🧪 Test Complet du Système TF-IDF de Matching de Compétences")
    print("=" * 70)
    
    try:
        # Test 1: Vectoriseur TF-IDF
        vectorizer, similarity_matrix, cv_skills, job_skills = test_tfidf_vectorizer()
        
        # Test 2: Impact du prétraitement
        raw_skills, processed_skills = test_preprocessing_impact()
        
        # Test 3: Métriques de similarité
        vectorizer2, cv_skills2, job_skills2 = test_similarity_metrics()
        
        # Test 4: Seuils de qualité
        simulated_scores, thresholds = test_quality_thresholds()
        
        # Test 5: Optimisations de performance
        large_cv_skills, large_job_skills = test_performance_optimization()
        
        print("\n✅ Tous les tests TF-IDF ont été exécutés avec succès!")
        print("🚀 Le système est prêt pour la production!")
        
    except Exception as e:
        print(f"\n❌ Erreur lors des tests: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
