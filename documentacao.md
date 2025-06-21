## Documentação: YouTube Comment Downloader com Análise de Sentimento BERT

### Visão Geral
Este programa permite coletar comentários de vídeos do YouTube, analisar seu sentimento usando o modelo BERT, e salvar/visualizar os resultados. Ele inclui uma interface gráfica (GUI) para interação com o usuário.

### Funcionalidades Principais
1. **Coleta de Comentários**:
   - Extrai comentários da API do YouTube
   - Suporta múltiplos vídeos simultaneamente
   - Configuração de limite máximo de comentários

2. **Análise de Sentimento**:
   - Utiliza o modelo BERT multilíngue (`nlptown/bert-base-multilingual-uncased-sentiment`)
   - Classifica sentimentos em escala de -1 (negativo) a 1 (positivo)
   - Processamento via GPU (se disponível)

3. **Filtragem e Ordenação**:
   - Filtra por número mínimo de likes
   - Ordena por likes, data ou sentimento

4. **Exportação de Dados**:
   - CSV
   - JSON
   - Banco de dados SQLite

5. **Visualização**:
   - Histograma de distribuição de sentimentos
   - Gráfico de pizza categorizado (negativo/neutro/positivo)

### Componentes-Chave

#### 1. Funções de Processamento
- **`get_video_id(url)`**: Extrai o ID do vídeo da URL do YouTube
- **`analyze_sentiment(comment)`**: 
  - Analisa sentimento usando BERT
  - Mapeia resultados de 1-5 estrelas para escala -1 a 1
  - Trunca textos > 512 caracteres
- **`get_comments(video_id)`**: 
  - Busca comentários via YouTube API
  - Processa até 1000 comentários/vídeo
- **`filter_comments()`**: Filtra por likes e sentimento
- **`sort_comments()`**: Ordena por likes/data/sentimento

#### 2. Funções de Exportação
- **`save_to_csv()`**: Salva em formato CSV
- **`save_to_json()`**: Salva em formato JSON
- **`save_to_database()`**: Armazena em SQLite

#### 3. Visualização
- **`plot_sentiment(comments)`**: 
  - Gera histograma + gráfico de pizza
  - Categoriza sentimentos em:
  ```python
  negativo: s < -0.3
  neutro: -0.3 <= s <= 0.3
  positivo: s > 0.3
  ```

#### 4. Interface Gráfica (Classe `YouTubeCommentDownloader`)
- **Campos de Entrada**:
  - URLs de vídeos (multilinha)
  - Máximo de comentários/vídeo
  - Mínimo de likes
  - ID do vídeo (para filtro de visualização)
  
- **Botões**:
  - *Baixar Comentários*: Inicia coleta e exportação
  - *Visualizar Sentimento*: Mostra gráficos

- **Indicadores**:
  - Barra de progresso
  - Status de operação

### Fluxo de Trabalho
1. Usuário insere URLs de vídeos
2. Define parâmetros (limite de comentários, filtro de likes)
3. Clica em "Baixar Comentários":
   - Comentários são coletados e analisados
   - Dados são salvos em CSV/JSON/SQLite
4. Para visualizar:
   - Insira ID do vídeo (opcional)
   - Clique em "Visualizar Sentimento"

### Requisitos de Execução
1. **Chave API YouTube**:
   - Substituir `API_KEY` por chave válida
   - Obtenha em: [Google Cloud Console](https://console.cloud.google.com/)

2. **Dependências**:
```bash
pip install google-api-python-client matplotlib pandas tkinter torch transformers
```

### Estrutura de Arquivos Gerados
Todos os arquivos são salvos na pasta `novo/`:
```
novo/
├── comentarios.csv
├── comentarios.json
└── comentarios.db (SQLite)
```

### Limitações
- Limite de 512 caracteres/comentário (restrição do BERT)
- Máximo ~1000 comentários/vídeo (limitação da API)
- Performance: Análise de sentimento pode ser lenta sem GPU

### Exemplo de Uso
```python
# Coletar comentários
comments = get_comments("dQw4w9WgXcQ", max_results=500)

# Filtrar e ordenar
filtered = filter_comments(comments, min_likes=5)
sorted_comments = sort_comments(filtered, sort_by='likes')

# Exportar e visualizar
save_to_csv(sorted_comments)
plot_sentiment(sorted_comments)
```