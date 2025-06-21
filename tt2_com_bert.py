import csv
import json
import logging
import sqlite3
from datetime import datetime
from tkinter import Tk, Label, Entry, Button, Text, END, filedialog, messagebox, ttk
import googleapiclient.discovery
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Configuração do logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Substitua pela sua chave de API
API_KEY = "AIzaSyDShBS7kjjhqvBWx885-Dv-Emj3s1Tzcv8"

# Inicialização do modelo BERT para análise de sentimentos
print("Carregando modelo BERT para análise de sentimentos...")
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    device=0 if torch.cuda.is_available() else -1
)
print("Modelo BERT carregado com sucesso!")

# Funções principais
def get_video_id(url):
    """Extrai o ID do vídeo da URL."""
    if 'v=' in url:
        return url.split('v=')[1].split('&')[0]
    return None

def analyze_sentiment(comment):
    """Analisa o sentimento de um comentário usando BERT."""
    try:
        # Limita o texto a 512 caracteres (limite do BERT)
        truncated_comment = comment[:512]

        # Faz a análise de sentimento
        result = sentiment_analyzer(truncated_comment)[0]

        # O modelo retorna labels de 1 a 5 estrelas
        # Convertemos para uma escala de -1 a 1
        label = result['label']
        score = result['score']

        # Mapeia as estrelas para polaridade
        star_to_polarity = {
            '1 star': -1.0,
            '2 stars': -0.5,
            '3 stars': 0.0,
            '4 stars': 0.5,
            '5 stars': 1.0
        }

        polarity = star_to_polarity.get(label, 0.0)

        # Ajusta a polaridade com base na confiança do modelo
        adjusted_polarity = polarity * 1

        return adjusted_polarity

    except Exception as e:
        logging.error(f"Erro na análise de sentimento: {e}")
        return 0.0  # Retorna neutro em caso de erro

def get_comments(video_id, max_results=1000):
    """Busca os comentários do vídeo."""
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=API_KEY)

    comments = []
    next_page_token = None
    total_results = 0

    while True:
        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                pageToken=next_page_token,
                textFormat="plainText",
                maxResults=min(100, max_results - total_results)
            )
            response = request.execute()

            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                sentiment = analyze_sentiment(comment['textDisplay'])
                comments.append({
                    'video_id': video_id,
                    'usuario': comment['authorDisplayName'],
                    'comentario': comment['textDisplay'],
                    'likes': comment['likeCount'],
                    'sentimento': sentiment,
                    'data': comment['publishedAt']
                })
                total_results += 1

                # Log de progresso a cada 10 comentários
                if total_results % 10 == 0:
                    logging.info(f"Processados {total_results} comentários...")

                if total_results >= max_results:
                    break

            next_page_token = response.get('nextPageToken')
            if not next_page_token or total_results >= max_results:
                break

        except Exception as e:
            logging.error(f"Erro ao buscar comentários: {e}")
            break

    return comments

def filter_comments(comments, min_likes=0, min_sentiment=-1, max_sentiment=1):
    """Filtra comentários com base em likes e sentimento."""
    return [
        comment for comment in comments
        if comment['likes'] >= min_likes and min_sentiment <= comment['sentimento'] <= max_sentiment
    ]

def sort_comments(comments, sort_by='likes', reverse=True):
    """Ordena os comentários."""
    if sort_by == 'likes':
        return sorted(comments, key=lambda x: x['likes'], reverse=reverse)
    elif sort_by == 'data':
        return sorted(comments, key=lambda x: x['data'], reverse=reverse)
    elif sort_by == 'sentimento':
        return sorted(comments, key=lambda x: x['sentimento'], reverse=reverse)
    return comments

def save_to_csv(comments, filename="novo/comentarios.csv"):
    """Salva os comentários em um arquivo CSV."""
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['video_id', 'usuario', 'comentario', 'likes', 'sentimento', 'data'])
        writer.writeheader()
        for comment in comments:
            writer.writerow(comment)
    logging.info(f"Comentários salvos em {filename}")

def save_to_json(comments, filename="novo/comentarios.json"):
    """Salva os comentários em um arquivo JSON."""
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, mode='w', encoding='utf-8') as file:
        json.dump(comments, file, ensure_ascii=False, indent=4)
    logging.info(f"Comentários salvos em {filename}")

def save_to_database(comments, db_file="novo/comentarios.db"):
    """Salva os comentários em um banco de dados SQLite."""
    import os
    os.makedirs(os.path.dirname(db_file), exist_ok=True)

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS comentarios (
            video_id TEXT,
            usuario TEXT,
            comentario TEXT,
            likes INTEGER,
            sentimento REAL,
            data TEXT
        )
    ''')
    for comment in comments:
        cursor.execute('''
            INSERT INTO comentarios (video_id, usuario, comentario, likes, sentimento, data)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (comment['video_id'], comment['usuario'], comment['comentario'], comment['likes'], comment['sentimento'], comment['data']))
    conn.commit()
    conn.close()
    logging.info(f"Comentários salvos no banco de dados {db_file}")

def plot_sentiment(comments):
    """Cria um gráfico de análise de sentimento."""
    sentiments = [comment['sentimento'] for comment in comments]

    plt.figure(figsize=(10, 6))

    # Histograma
    plt.subplot(1, 2, 1)
    plt.hist(sentiments, bins=20, color='blue', alpha=0.7)
    plt.title("Distribuição de Sentimento dos Comentários")
    plt.xlabel("Sentimento (-1 a 1)")
    plt.ylabel("Frequência")

    # Gráfico de pizza para categorias
    plt.subplot(1, 2, 2)
    negative = sum(1 for s in sentiments if s < -0.3)
    neutral = sum(1 for s in sentiments if -0.3 <= s <= 0.3)
    positive = sum(1 for s in sentiments if s > 0.3)

    labels = ['Negativo', 'Neutro', 'Positivo']
    sizes = [negative, neutral, positive]
    colors = ['red', 'gray', 'green']

    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
    plt.title("Distribuição por Categoria de Sentimento")

    plt.tight_layout()
    plt.show()

# Interface Gráfica
class YouTubeCommentDownloader:
    def __init__(self, root):
        self.root = root
        self.root.title("YouTube Comment Downloader com BERT")
        self.root.geometry("700x500")

        # Widgets
        Label(root, text="URLs dos Vídeos (uma por linha):", font=("Arial", 12)).pack(pady=5)
        self.url_entry = Text(root, height=5, width=70)
        self.url_entry.pack(pady=5)

        Label(root, text="Número Máximo de Comentários por Vídeo:", font=("Arial", 10)).pack(pady=5)
        self.max_results_entry = Entry(root)
        self.max_results_entry.pack(pady=5)
        self.max_results_entry.insert(0, "100")

        Label(root, text="Filtrar por Mínimo de Likes:", font=("Arial", 10)).pack(pady=5)
        self.min_likes_entry = Entry(root)
        self.min_likes_entry.pack(pady=5)
        self.min_likes_entry.insert(0, "0")

        # Frame para botões
        button_frame = ttk.Frame(root)
        button_frame.pack(pady=20)

        Button(button_frame, text="Baixar Comentários", command=self.download_comments,
               bg="green", fg="white", font=("Arial", 10)).pack(side="left", padx=5)
        Button(button_frame, text="Visualizar Sentimento", command=self.show_sentiment_plot,
               bg="blue", fg="white", font=("Arial", 10)).pack(side="left", padx=5)
        
        Label(root, text="Id do video", font=("Arial", 10)).pack(pady=5)
        self.video_id_entry = Entry(root, font=("Arial", 10), width=40)
        self.video_id_entry.pack(pady=5)
        

        # Status label
        self.status_label = Label(root, text="Pronto para usar", font=("Arial", 10), fg="green")
        self.status_label.pack(pady=10)

        # Progress bar
        self.progress = ttk.Progressbar(root, length=400, mode='indeterminate')
        self.progress.pack(pady=10)

    def download_comments(self):
        """Baixa os comentários e salva em CSV, JSON e banco de dados."""
        self.status_label.config(text="Processando...", fg="orange")
        self.progress.start()

        urls = self.url_entry.get("1.0", END).strip().split('\n')
        max_results = int(self.max_results_entry.get())
        min_likes = int(self.min_likes_entry.get())

        all_comments = []
        for url in urls:
            video_id = get_video_id(url)
            if not video_id:
                messagebox.showerror("Erro", f"URL inválida: {url}")
                continue

            logging.info(f"Baixando comentários do vídeo: {video_id}")
            self.status_label.config(text=f"Baixando comentários do vídeo: {video_id}")
            self.root.update()

            comments = get_comments(video_id, max_results)
            comments = filter_comments(comments, min_likes=min_likes)
            all_comments.extend(comments)

        self.progress.stop()

        if all_comments:
            save_to_csv(all_comments)
            save_to_json(all_comments)
            save_to_database(all_comments)
            self.status_label.config(text="Comentários baixados com sucesso!", fg="green")
            messagebox.showinfo("Sucesso", f"Total de {len(all_comments)} comentários baixados e salvos!")
        else:
            self.status_label.config(text="Nenhum comentário baixado", fg="red")
            messagebox.showwarning("Aviso", "Nenhum comentário foi baixado.")

    def show_sentiment_plot(self):
        """Exibe o gráfico de análise de sentimento."""
        try:
            conn = sqlite3.connect("novo/comentarios.db")
            cursor = conn.cursor()
            sentiments = []
            #quero pegar o video_id_entry
            video_id = self.video_id_entry.get()
            if not video_id:
                
                cursor.execute("SELECT sentimento FROM comentarios")
                sentiments = [row[0] for row in cursor.fetchall()]
                conn.close()
            else:
                #filtra com id_video
                cursor.execute("SELECT sentimento FROM comentarios WHERE video_id = ?", (video_id,))
                sentiments = [row[0] for row in cursor.fetchall()]
                conn.close()         

            if sentiments:
                plot_sentiment([{'sentimento': s} for s in sentiments])
            else:
                messagebox.showwarning("Aviso", "Nenhum dado de sentimento encontrado.")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar dados: {e}")

# Executar a aplicação
if __name__ == "__main__":
    root = Tk()
    app = YouTubeCommentDownloader(root)
    root.mainloop()