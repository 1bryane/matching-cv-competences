#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 Prétraitement Avancé des Compétences pour TF-IDF

Ce module fournit des fonctions avancées pour le prétraitement des compétences
avant leur utilisation dans le vectoriseur TF-IDF.
"""

import re
import string
from typing import List, Dict, Set
from tfidf_config import get_synonyms_config, get_preprocessing_config

class SkillsPreprocessor:
    """
    Classe pour le prétraitement avancé des compétences
    """
    
    def __init__(self):
        """Initialise le prétraiteur avec la configuration"""
        self.synonyms_config = get_synonyms_config()
        self.preprocessing_config = get_preprocessing_config()
        self._build_synonyms_mapping()
    
    def _build_synonyms_mapping(self):
        """Construit le mapping des synonymes"""
        self.synonyms_mapping = {}
        
        # Ajout des synonymes par catégorie
        for category, synonyms in self.synonyms_config.items():
            for short_form, full_form in synonyms.items():
                self.synonyms_mapping[short_form.lower()] = full_form.lower()
                self.synonyms_mapping[full_form.lower()] = full_form.lower()  # Normalisation
    
    def clean_text(self, text: str) -> str:
        """
        Nettoie le texte des compétences
        
        Args:
            text (str): Texte brut des compétences
            
        Returns:
            str: Texte nettoyé
        """
        if not text:
            return ""
        
        # Conversion en minuscules
        if self.preprocessing_config['normalize_case']:
            text = text.lower()
        
        # Remplacement des séparateurs multiples par des virgules
        separators = self.preprocessing_config['separators']
        for sep in separators:
            text = text.replace(sep, ',')
        
        # Suppression des espaces multiples
        text = re.sub(r'\s+', ' ', text)
        
        # Suppression des caractères spéciaux non utiles
        if self.preprocessing_config['remove_special_chars']:
            # Garder les lettres, chiffres, espaces, virgules et points
            text = re.sub(r'[^\w\s,.-]', '', text)
        
        # Nettoyage des espaces autour des virgules
        text = re.sub(r'\s*,\s*', ', ', text)
        
        # Suppression des espaces en début et fin
        text = text.strip()
        
        return text
    
    def expand_synonyms(self, text: str) -> str:
        """
        Développe les synonymes et acronymes dans le texte
        
        Args:
            text (str): Texte des compétences
            
        Returns:
            str: Texte avec synonymes développés
        """
        if not text:
            return ""
        
        # Division en tokens
        tokens = text.split(',')
        expanded_tokens = []
        
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            
            # Vérification des synonymes
            expanded_token = self.synonyms_mapping.get(token.lower(), token)
            expanded_tokens.append(expanded_token)
        
        return ', '.join(expanded_tokens)
    
    def normalize_technical_terms(self, text: str) -> str:
        """
        Normalise les termes techniques courants
        
        Args:
            text (str): Texte des compétences
            
        Returns:
            str: Texte avec termes normalisés
        """
        if not text:
            return ""
        
        # Normalisations courantes
        normalizations = {
            'api': 'apis',
            'ui': 'user interface',
            'ux': 'user experience',
            'db': 'database',
            'sql': 'sql database',
            'nosql': 'nosql database',
            'ml': 'machine learning',
            'ai': 'artificial intelligence',
            'dl': 'deep learning',
            'ds': 'data science',
            'devops': 'devops practices',
            'ci/cd': 'continuous integration continuous deployment',
            'aws': 'amazon web services',
            'azure': 'microsoft azure',
            'gcp': 'google cloud platform',
            'k8s': 'kubernetes',
            'docker': 'docker containerization',
            'git': 'version control',
            'agile': 'agile methodology',
            'scrum': 'scrum framework'
        }
        
        normalized_text = text
        for short_form, full_form in normalizations.items():
            # Remplacement avec respect de la casse
            pattern = re.compile(re.escape(short_form), re.IGNORECASE)
            normalized_text = pattern.sub(full_form, normalized_text)
        
        return normalized_text
    
    def remove_noise_words(self, text: str) -> str:
        """
        Supprime les mots de bruit techniques
        
        Args:
            text (str): Texte des compétences
            
        Returns:
            str: Texte sans mots de bruit
        """
        if not text:
            return ""
        
        # Mots de bruit techniques
        noise_words = {
            'version', 'release', 'update', 'patch', 'bug', 'fix', 'issue',
            'feature', 'tool', 'software', 'application', 'system', 'platform',
            'technology', 'framework', 'library', 'package', 'module', 'component'
        }
        
        # Division en tokens
        tokens = text.split(',')
        filtered_tokens = []
        
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            
            # Vérification si le token n'est pas un mot de bruit
            if token.lower() not in noise_words:
                filtered_tokens.append(token)
        
        return ', '.join(filtered_tokens)
    
    def preprocess_skills(self, skills_text: str) -> str:
        """
        Prétraite complètement le texte des compétences
        
        Args:
            skills_text (str): Texte brut des compétences
            
        Returns:
            str: Texte prétraité et optimisé
        """
        if not skills_text:
            return ""
        
        # Étapes de prétraitement
        text = self.clean_text(skills_text)
        text = self.expand_synonyms(text)
        text = self.normalize_technical_terms(text)
        text = self.remove_noise_words(text)
        
        # Nettoyage final
        text = self.clean_text(text)
        
        return text
    
    def preprocess_skills_batch(self, skills_list: List[str]) -> List[str]:
        """
        Prétraite une liste de compétences
        
        Args:
            skills_list (List[str]): Liste des textes de compétences
            
        Returns:
            List[str]: Liste des compétences prétraitées
        """
        return [self.preprocess_skills(skills) for skills in skills_list]
    
    def get_skills_statistics(self, skills_list: List[str]) -> Dict:
        """
        Calcule des statistiques sur les compétences
        
        Args:
            skills_list (List[str]): Liste des compétences
            
        Returns:
            Dict: Statistiques des compétences
        """
        if not skills_list:
            return {}
        
        # Compétences prétraitées
        processed_skills = self.preprocess_skills_batch(skills_list)
        
        # Statistiques
        stats = {
            'total_skills': len(skills_list),
            'total_tokens': 0,
            'unique_tokens': set(),
            'avg_tokens_per_skill': 0,
            'most_common_tokens': {},
            'synonyms_expanded': 0
        }
        
        for original, processed in zip(skills_list, processed_skills):
            original_tokens = len(original.split(','))
            processed_tokens = len(processed.split(','))
            
            stats['total_tokens'] += processed_tokens
            stats['unique_tokens'].update(processed.split(','))
            
            # Comptage des synonymes développés
            if processed_tokens > original_tokens:
                stats['synonyms_expanded'] += 1
        
        stats['unique_tokens'] = len(stats['unique_tokens'])
        stats['avg_tokens_per_skill'] = stats['total_tokens'] / stats['total_skills']
        
        return stats

def create_preprocessor() -> SkillsPreprocessor:
    """Crée et retourne une instance du prétraiteur"""
    return SkillsPreprocessor()

if __name__ == "__main__":
    # Test du prétraiteur
    preprocessor = create_preprocessor()
    
    # Exemples de test
    test_skills = [
        "Python, ML, Deep Learning, TensorFlow",
        "Java, Spring, REST APIs, Microservices",
        "JS, React, Node.js, MongoDB",
        "DevOps, Docker, K8s, CI/CD, AWS"
    ]
    
    print("🧪 Test du prétraiteur de compétences")
    print("=" * 50)
    
    for i, skills in enumerate(test_skills):
        processed = preprocessor.preprocess_skills(skills)
        print(f"\nCompétences {i+1}:")
        print(f"  Original: {skills}")
        print(f"  Traité:   {processed}")
    
    # Statistiques
    stats = preprocessor.get_skills_statistics(test_skills)
    print(f"\n📊 Statistiques:")
    print(f"  Total compétences: {stats['total_skills']}")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Tokens uniques: {stats['unique_tokens']}")
    print(f"  Moyenne tokens/compétence: {stats['avg_tokens_per_skill']:.1f}")
    print(f"  Synonymes développés: {stats['synonyms_expanded']}")
