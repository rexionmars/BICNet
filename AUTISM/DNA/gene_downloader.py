import pandas as pd
import numpy as np
from Bio import SeqIO, Entrez
from Bio.Seq import Seq
from Bio.SeqUtils import GC
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import os
from ftplib import FTP

class GeneticDataDownloader:
    def __init__(self, email):
        """
        Inicializa o downloader com email do pesquisador para acesso às bases de dados
        """
        Entrez.email = email
        self.credentials = None
        
    def set_database_credentials(self, username, password, database_url):
        """
        Configure credenciais para bases de dados que requerem autenticação
        """
        self.credentials = {
            'username': username,
            'password': password,
            'database_url': database_url
        }
    
    def download_from_ncbi(self, accession_id, output_file):
        """
        Download de sequências do NCBI usando ID de acesso
        """
        try:
            handle = Entrez.efetch(db="nucleotide", 
                                 id=accession_id, 
                                 rettype="fasta", 
                                 retmode="text")
            with open(output_file, 'w') as out_handle:
                out_handle.write(handle.read())
            return f"Sequência {accession_id} baixada com sucesso"
        except Exception as e:
            return f"Erro no download: {str(e)}"
    
    def download_from_protected_database(self, sample_id, output_file):
        """
        Download de banco de dados protegido usando credenciais
        """
        if not self.credentials:
            return "Credenciais não configuradas"
            
        try:
            session = requests.Session()
            # Autenticação
            auth_response = session.post(
                f"{self.credentials['database_url']}/auth",
                json={
                    'username': self.credentials['username'],
                    'password': self.credentials['password']
                }
            )
            
            if auth_response.status_code != 200:
                return "Falha na autenticação"
                
            # Download dos dados
            response = session.get(
                f"{self.credentials['database_url']}/download/{sample_id}",
                stream=True
            )
            
            if response.status_code == 200:
                with open(output_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return f"Arquivo baixado com sucesso: {output_file}"
            else:
                return f"Erro no download: {response.status_code}"
                
        except Exception as e:
            return f"Erro durante o download: {str(e)}"
    
    def batch_download(self, id_list, output_dir):
        """
        Download em lote de múltiplas sequências
        """
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        for sample_id in id_list:
            output_file = os.path.join(output_dir, f"{sample_id}.fasta")
            result = self.download_from_protected_database(sample_id, output_file)
            results.append((sample_id, result))
            
        return results

class GeneticAnalyzer:
    def __init__(self):
        self.sequences = {}
        self.analysis_results = {}
        
    # [Todos os métodos anteriores permanecem iguais]
    def load_fasta(self, file_path):
        try:
            for record in SeqIO.parse(file_path, "fasta"):
                self.sequences[record.id] = str(record.seq)
            return f"Carregadas {len(self.sequences)} sequências"
        except Exception as e:
            return f"Erro ao carregar arquivo: {str(e)}"
    
    def analyze_sequence(self, seq_id):
        if seq_id not in self.sequences:
            return "Sequência não encontrada"
        
        seq = self.sequences[seq_id]
        analysis = {
            'length': len(seq),
            'gc_content': GC(seq),
            'base_counts': {
                'A': seq.count('A'),
                'T': seq.count('T'),
                'G': seq.count('G'),
                'C': seq.count('C')
            }
        }
        self.analysis_results[seq_id] = analysis
        return analysis

# Exemplo de uso
def main():
    # Configurar downloader
    downloader = GeneticDataDownloader("seu_email@instituicao.edu")
    downloader.set_database_credentials(
        username="seu_usuario",
        password="sua_senha",
        database_url="https://sua_base_dados.edu"
    )
    
    # Download de dados
    sample_ids = ["amostra1", "amostra2", "amostra3"]
    download_results = downloader.batch_download(sample_ids, "dados_download")
    
    # Análise dos dados baixados
    analyzer = GeneticAnalyzer()
    
    # Processar cada arquivo baixado
    for sample_id in sample_ids:
        file_path = f"dados_download/{sample_id}.fasta"
        if os.path.exists(file_path):
            analyzer.load_fasta(file_path)
            analysis = analyzer.analyze_sequence(sample_id)
            print(f"Análise da amostra {sample_id}:", analysis)

if __name__ == "__main__":
    main()