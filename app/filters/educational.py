from sklearn.feature_extraction.text import TfidfVectorizer
from .base import BaseFilter
from typing import Dict, Any

class EducationalFilter(BaseFilter):
    """
    Filtro para classificar vídeos por conteúdo educacional.
    """
    
    def __init__(self):
        print("EducationalFilter :: __init__()")
        super().__init__(
            name="Educacional",
            description="Filtra por conteúdo educacional",
            default_enabled=True
        )
        self.name = "Educacional"
        self.description = "Filtra por conteúdo educacional"
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        # Categorias educacionais e seus termos relacionados
        self.categories = {
            "ciências": ["física", "química", "biologia", "ciência", "experimento", "laboratório",
                        "natureza", "planeta", "animal", "planta", "corpo", "saúde"],
            "história": ["história", "passado", "civilização", "guerra", "revolução", "império",
                        "cultura", "sociedade", "país", "mundo", "descoberta"],
            "matemática": ["matemática", "número", "conta", "geometria", "álgebra", "cálculo",
                          "soma", "subtração", "multiplicação", "divisão", "forma"],
            "artes": ["arte", "música", "pintura", "desenho", "teatro", "dança", "criatividade",
                     "imaginação", "cor", "som", "instrumento"],
            "linguagem": ["português", "palavra", "letra", "gramática", "leitura", "escrita",
                         "história", "livro", "poesia", "comunicação", "expressão"],
            "tecnologia": ["computador", "internet", "programação", "robô", "tecnologia",
                          "digital", "aplicativo", "software", "inovação", "futuro"],
            "cidadania": ["cidadania", "direito", "dever", "respeito", "sociedade",
                         "comunidade", "cooperação", "ajuda", "amizade", "família"]
        }
        
    def process(self, video_data: Dict[str, Any]) -> float:
        print("EducationalFilter :: process()")
        """
        Processa o vídeo e retorna uma pontuação entre 0 e 1.
        1 = mais educacional, 0 = menos educacional
        """
        text = f"{video_data.get('title', '')} {video_data.get('description', '')} {video_data.get('transcript', '')}"
        if not text.strip():
            return 0.0
            
        try:
            # Calcula TF-IDF do texto
            tfidf_matrix = self.vectorizer.fit_transform([text])
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Conta quantos termos educacionais foram encontrados
            educational_terms = 0
            total_categories = len(self.categories)
            categories_found = set()
            
            for word in feature_names:
                for category, terms in self.categories.items():
                    if word in terms:
                        educational_terms += 1
                        categories_found.add(category)
                        
            # Calcula o score baseado na quantidade de termos e categorias encontradas
            term_score = min(1.0, educational_terms / 10)  # Normaliza para máximo de 10 termos
            category_score = len(categories_found) / total_categories
            
            # Média ponderada: 60% termos, 40% categorias
            final_score = (0.6 * term_score) + (0.4 * category_score)
            return final_score
            
        except Exception as e:
            print(f"Erro ao processar texto educacional: {e}")
            return 0.0

    def get_filter_info(self) -> Dict[str, Any]:
        print("EducationalFilter :: get_filter_info()")
        return {
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "type": "educational",
            "default_value": 100,
            "options": {
                "educational_levels": [
                    {"value": "all", "label": "Todos os níveis"},
                    {"value": "basic", "label": "Básico"},
                    {"value": "intermediate", "label": "Intermediário"},
                    {"value": "advanced", "label": "Avançado"}
                ]
            }
        } 