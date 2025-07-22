from Bio import Entrez, SeqIO
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os

def download_rat_gene(gene_id):
    """
    Baixa uma sequência de DNA de rato do NCBI usando o ID do gene
    """
    Entrez.email = "leonardimelo43@gmail.com"
    
    try:
        handle = Entrez.efetch(db="nucleotide", id=gene_id, rettype="fasta", retmode="text")
        record = SeqIO.read(handle, "fasta")
        handle.close()
        return record
    except Exception as e:
        print(f"Erro ao baixar sequência: {e}")
        return None

def save_sequence(sequence, output_dir="dna_files"):
    """
    Salva a sequência em múltiplos formatos
    """
    # Cria o diretório se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Base do nome do arquivo (usando o ID da sequência)
    base_filename = os.path.join(output_dir, sequence.id.replace(".", "_"))
    
    # Salva em formato FASTA
    fasta_file = f"{base_filename}.fasta"
    SeqIO.write(sequence, fasta_file, "fasta")
    
    # Salva em formato texto simples
    txt_file = f"{base_filename}.txt"
    with open(txt_file, "w") as f:
        f.write(str(sequence.seq))
    
    # Salva informações em CSV
    csv_file = f"{base_filename}_info.csv"
    seq_info = {
        "ID": [sequence.id],
        "Descrição": [sequence.description],
        "Comprimento": [len(sequence.seq)]
    }
    pd.DataFrame(seq_info).to_csv(csv_file, index=False)
    
    return {
        "fasta": fasta_file,
        "txt": txt_file,
        "csv": csv_file
    }

def analyze_sequence(sequence):
    """
    Realiza análises básicas na sequência de DNA
    """
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

def plot_base_distribution(base_counts, output_dir="dna_files"):
    """
    Cria e salva um gráfico de barras da distribuição das bases
    """
    bases = list(base_counts.keys())
    counts = list(base_counts.values())
    
    plt.figure(figsize=(8, 6))
    plt.bar(bases, counts)
    plt.title('Distribuição das Bases no DNA')
    plt.xlabel('Base')
    plt.ylabel('Frequência')
    
    # Salva o gráfico
    os.makedirs(output_dir, exist_ok=True)
    plot_file = os.path.join(output_dir, "base_distribution.png")
    plt.savefig(plot_file)
    plt.close()
    
    return plot_file

def main():
    # ID de exemplo de um gene de rato (BDNF)
    gene_id = "NM_012513"
    
    # Baixa a sequência
    sequence = download_rat_gene(gene_id)
    
    if sequence:
        print(f"Sequência baixada: {sequence.id}")
        print(f"Descrição: {sequence.description}\n")
        
        # Salva a sequência
        saved_files = save_sequence(sequence)
        print("\nArquivos salvos:")
        for format, filepath in saved_files.items():
            print(f"{format.upper()}: {filepath}")
        
        # Analisa a sequência
        results = analyze_sequence(sequence)
        
        # Mostra os resultados
        print("\nResultados da análise:")
        print(f"Tamanho da sequência: {results['tamanho']} bases")
        print(f"Contagem de bases: {results['contagem_bases']}")
        print(f"Conteúdo GC: {results['conteudo_gc']:.2f}%")
        print("\nMotivos mais comuns de 4 bases:")
        for motif, count in results['motivos_comuns'].items():
            print(f"{motif}: {count} ocorrências")
        
        # Plota e salva a distribuição das bases
        plot_file = plot_base_distribution(results['contagem_bases'])
        print(f"\nGráfico salvo em: {plot_file}")

if __name__ == "__main__":
    main()