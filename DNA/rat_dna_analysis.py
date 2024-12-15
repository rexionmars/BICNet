from Bio import Entrez, SeqIO
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def download_rat_gene(gene_id):
    """
    Baixa uma sequência de DNA de rato do NCBI usando o ID do gene
    """
    # Sempre forneça seu email ao usar o Entrez
    Entrez.email = "leonardimelo43@gmail.com"
    
    try:
        # Baixa a sequência do NCBI
        handle = Entrez.efetch(db="nucleotide", id=gene_id, rettype="fasta", retmode="text")
        record = SeqIO.read(handle, "fasta")
        handle.close()
        return record
    except Exception as e:
        print(f"Erro ao baixar sequência: {e}")
        return None

def analyze_sequence(sequence):
    """
    Realiza análises básicas na sequência de DNA
    """
    # Converte para string e maiúsculas
    seq_str = str(sequence.seq).upper()
    
    # Conta a frequência de cada base
    base_counts = Counter(seq_str)
    
    # Calcula o conteúdo GC
    gc_content = ((base_counts['G'] + base_counts['C']) / len(seq_str)) * 100
    
    # Encontra motivos comuns de 4 bases
    motifs = {}
    for i in range(len(seq_str) - 3):
        motif = seq_str[i:i+4]
        motifs[motif] = motifs.get(motif, 0) + 1
    
    return {
        'tamanho': len(seq_str),
        'contagem_bases': dict(base_counts),
        'conteudo_gc': gc_content,
        'motivos_comuns': dict(sorted(motifs.items(), key=lambda x: x[1], reverse=True)[:5])
    }

def plot_base_distribution(base_counts):
    """
    Cria um gráfico de barras da distribuição das bases
    """
    bases = list(base_counts.keys())
    counts = list(base_counts.values())
    
    plt.figure(figsize=(8, 6))
    plt.bar(bases, counts)
    plt.title('Distribuição das Bases no DNA')
    plt.xlabel('Base')
    plt.ylabel('Frequência')
    # save the plot
    plt.savefig('base_distribution.png')

# Exemplo de uso
def main():
    # ID de exemplo de um gene de rato (BDNF - Brain-derived neurotrophic factor)
    gene_id = "NM_012513"
    
    # Baixa a sequência
    sequence = download_rat_gene(gene_id)
    
    if sequence:
        print(f"Sequência baixada: {sequence.id}")
        print(f"Descrição: {sequence.description}\n")
        
        # Analisa a sequência
        results = analyze_sequence(sequence)
        
        # Mostra os resultados
        print("Resultados da análise:")
        print(f"Tamanho da sequência: {results['tamanho']} bases")
        print(f"Contagem de bases: {results['contagem_bases']}")
        print(f"Conteúdo GC: {results['conteudo_gc']:.2f}%")
        print("\nMotivos mais comuns de 4 bases:")
        for motif, count in results['motivos_comuns'].items():
            print(f"{motif}: {count} ocorrências")
        
        # Plota a distribuição das bases
        plot_base_distribution(results['contagem_bases'])

if __name__ == "__main__":
    main()