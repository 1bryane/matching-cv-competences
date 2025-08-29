#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 Démonstration du Système TF-IDF de Matching de Compétences

Ce script démontre les capacités du système TF-IDF optimisé
pour le matching de compétences entre CVs et offres d'emploi.
"""

from skills_preprocessor import create_preprocessor
from tfidf_config import get_tfidf_config, get_quality_thresholds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def demo_basic_matching():
    """Démonstration du matching de base avec TF-IDF"""
    print("🎯 Démonstration du Matching TF-IDF de Compétences")
    print("=" * 60)
    
    # Données de démonstration
    cv_skills = [
        "Python, Data Science, Machine Learning, Deep Learning, TensorFlow, Keras",
        "Java, Spring Boot, REST APIs, Microservices, Hibernate, Maven",
        "JavaScript, React, Redux, Node.js, Express, MongoDB, TypeScript",
        "DevOps, Docker, Kubernetes, CI/CD, Jenkins, Git, AWS, Azure",
        "Project Management, Scrum, Agile, JIRA, Confluence, Risk Management"
    ]
    
    job_skills = [
        "Python, Machine Learning, Deep Learning, Neural Networks, Data Science",
        "Java, Spring Boot, Microservices, Cloud Native, Docker, Kubernetes",
        "JavaScript, React, TypeScript, Modern Web Development, REST APIs",
        "DevOps Engineer, Containerization, CI/CD, Cloud Infrastructure, Monitoring"
    ]
    
    print(f"📊 {len(cv_skills)} CVs et {len(job_skills)} offres d'emploi")
    
    # Prétraitement des compétences
    preprocessor = create_preprocessor()
    cv_processed = preprocessor.preprocess_skills_batch(cv_skills)
    job_processed = preprocessor.preprocess_skills_batch(job_skills)
    
    print("\n🔧 Compétences après prétraitement:")
    for i, (original, processed) in enumerate(zip(cv_skills, cv_processed)):
        print(f"  CV {i+1}: {original}")
        print(f"       → {processed}")
        print()
    
    # Configuration TF-IDF
    config = get_tfidf_config()
    vectorizer = TfidfVectorizer(**config)
    
    # Vectorisation
    all_skills = cv_processed + job_processed
    tfidf_matrix = vectorizer.fit_transform(all_skills)
    
    # Séparation des vecteurs
    cv_vectors = tfidf_matrix[:len(cv_processed)]
    job_vectors = tfidf_matrix[len(cv_processed):]
    
    # Calcul de similarité
    similarity_matrix = cosine_similarity(cv_vectors, job_vectors)
    
    print(f"✅ Matching TF-IDF calculé!")
    print(f"📊 Matrice de similarité: {similarity_matrix.shape}")
    
    # Affichage des résultats
    print("\n🏆 Meilleurs matches:")
    for job_idx in range(len(job_skills)):
        scores = similarity_matrix[:, job_idx]
        best_cv_idx = np.argmax(scores)
        best_score = scores[best_cv_idx]
        
        print(f"\nJob {job_idx+1}: {job_skills[job_idx]}")
        print(f"  → CV {best_cv_idx+1}: {cv_skills[best_cv_idx]}")
        print(f"  Score: {best_score:.4f}")
        
        # Top 3 CVs pour ce job
        top_cvs = np.argsort(scores)[::-1][:3]
        print(f"  Top 3 CVs:")
        for rank, cv_idx in enumerate(top_cvs):
            print(f"    {rank+1}. CV {cv_idx+1}: {scores[cv_idx]:.4f}")
    
    return similarity_matrix, cv_skills, job_skills

def demo_preprocessing_features():
    """Démonstration des fonctionnalités de prétraitement"""
    print("\n🔧 Démonstration des Fonctionnalités de Prétraitement")
    print("=" * 60)
    
    preprocessor = create_preprocessor()
    
    # Exemples de compétences avec différents formats
    test_skills = [
        "Python, ML, Deep Learning, TensorFlow",
        "Java, Spring, REST APIs, Microservices",
        "JS, React, Node.js, MongoDB",
        "DevOps, Docker, K8s, CI/CD, AWS",
        "Project Mgmt, Scrum, Agile, JIRA"
    ]
    
    print("📝 Exemples de prétraitement:")
    for i, skills in enumerate(test_skills):
        processed = preprocessor.preprocess_skills(skills)
        print(f"\nCompétences {i+1}:")
        print(f"  Original: {skills}")
        print(f"  Traité:   {processed}")
    
    # Statistiques
    stats = preprocessor.get_skills_statistics(test_skills)
    print(f"\n📊 Statistiques du prétraitement:")
    print(f"  Total compétences: {stats['total_skills']}")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Tokens uniques: {stats['unique_tokens']}")
    print(f"  Synonymes développés: {stats['synonyms_expanded']}")

def demo_quality_analysis():
    """Démonstration de l'analyse de qualité des matches"""
    print("\n🎯 Démonstration de l'Analyse de Qualité")
    print("=" * 60)
    
    # Génération de scores simulés
    np.random.seed(42)
    simulated_scores = np.random.uniform(0, 1, 50)
    
    # Seuils de qualité
    thresholds = get_quality_thresholds()
    
    print(f"📊 Seuils de qualité configurés:")
    for key, value in thresholds.items():
        print(f"  {key}: {value}")
    
    print(f"\n📈 Analyse de {len(simulated_scores)} scores simulés:")
    print(f"  Score min/max: {simulated_scores.min():.4f} / {simulated_scores.max():.4f}")
    print(f"  Score moyen: {simulated_scores.mean():.4f}")
    print(f"  Écart-type: {simulated_scores.std():.4f}")
    
    # Répartition par qualité
    high_quality = np.sum(simulated_scores > thresholds['high_quality'])
    medium_quality = np.sum((simulated_scores > thresholds['medium_quality']) & 
                           (simulated_scores <= thresholds['high_quality']))
    low_quality = np.sum(simulated_scores <= thresholds['medium_quality'])
    
    print(f"\n🎯 Répartition par qualité:")
    print(f"  Haute qualité (>{thresholds['high_quality']}): {high_quality} ({high_quality/len(simulated_scores)*100:.1f}%)")
    print(f"  Qualité moyenne ({thresholds['medium_quality']}-{thresholds['high_quality']}): {medium_quality} ({medium_quality/len(simulated_scores)*100:.1f}%)")
    print(f"  Faible qualité (≤{thresholds['medium_quality']}): {low_quality} ({low_quality/len(simulated_scores)*100:.1f}%)")

def demo_configuration():
    """Démonstration de la configuration TF-IDF"""
    print("\n⚙️ Démonstration de la Configuration TF-IDF")
    print("=" * 60)
    
    # Configuration TF-IDF
    tfidf_config = get_tfidf_config()
    print(f"📊 Configuration TF-IDF ({len(tfidf_config)} paramètres):")
    for key, value in tfidf_config.items():
        print(f"  {key}: {value}")
    
    # Seuils de qualité
    quality_thresholds = get_quality_thresholds()
    print(f"\n🎯 Seuils de qualité ({len(quality_thresholds)} seuils):")
    for key, value in quality_thresholds.items():
        print(f"  {key}: {value}")
    
    print(f"\n✅ Configuration chargée avec succès!")
    print(f"🚀 Le système est prêt pour le matching de compétences!")

def main():
    """Fonction principale de démonstration"""
    try:
        # Démonstration 1: Matching de base
        similarity_matrix, cv_skills, job_skills = demo_basic_matching()
        
        # Démonstration 2: Fonctionnalités de prétraitement
        demo_preprocessing_features()
        
        # Démonstration 3: Analyse de qualité
        demo_quality_analysis()
        
        # Démonstration 4: Configuration
        demo_configuration()
        
        print("\n🎉 Démonstration terminée avec succès!")
        print("🚀 Le système TF-IDF est prêt pour la production!")
        
    except Exception as e:
        print(f"\n❌ Erreur lors de la démonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
