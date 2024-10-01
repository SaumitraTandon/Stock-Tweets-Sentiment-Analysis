# Stock Tweets Sentiment Analysis

This project performs sentiment analysis on stock-related tweets to determine the general sentiment (positive or negative) of discussions around specific stocks or market trends. The analysis includes:
- **Sentiment polarity analysis**: Determines whether discussions are positive or negative.
- **Frequency of mentions**: Counts how often specific stocks or market trends are mentioned.
- **Topic modeling**: Identifies key discussion themes through natural language processing techniques.

## Features

- **Sentiment Analysis**: Classifies tweets into positive or negative sentiments based on pre-labeled data using sentiment analysis models such as VADER.
- **Stock Mention Frequency**: Analyzes the frequency of stock mentions by filtering tweets containing specific stock symbols (e.g., $AAPL, $TSLA).
- **Topic Modeling**: Uses Latent Dirichlet Allocation (LDA) or similar topic modeling techniques to identify common themes or topics being discussed in relation to stocks.
- **Data Scraping**: Allows real-time scraping of tweets using the Twitter API (`tweepy`), which can then be analyzed for sentiment, stock mentions, and topics.

## Dataset

The dataset used for this analysis consists of tweets collected between April 2020 and July 2020. Each tweet contains the following fields:
- `id`: Unique identifier for each tweet.
- `created_at`: Timestamp of when the tweet was posted.
- `text`: Content of the tweet.
- `sentiment`: Pre-labeled sentiment (`positive` or `negative`).

You can also use the data scraping module to collect real-time tweets on specific stocks and trends.

## Installation

To run this project locally, ensure you have Python 3.x installed and follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/stock-tweets-sentiment-analysis.git
   ```
   
2. Navigate to the project directory:
   ```bash
   cd stock-tweets-sentiment-analysis
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   **Dependencies**:
   - `numpy`
   - `pandas`
   - `scikit-learn`
   - `matplotlib`
   - `nltk` (for natural language processing tasks)
   - `tweepy` (for Twitter API access, if needed for data scraping)
   - `gensim` (for topic modeling)

4. (Optional) Download NLTK datasets if necessary:
   ```python
   import nltk
   nltk.download('vader_lexicon')  # For sentiment analysis
   ```

## Usage

### Sentiment Analysis
Run the sentiment analysis notebook to process the dataset and generate sentiment polarity scores:
1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Stock_Tweets_Sentiment_Analysis.ipynb
   ```

2. Execute the cells to:
   - Load and clean the dataset.
   - Perform sentiment analysis using VADER or another sentiment analysis tool.
   - Visualize sentiment distributions.

### Frequency of Stock Mentions

In the notebook, you can analyze how frequently specific stocks are mentioned. This can be done by:
1. Filtering the dataset for specific stock symbols (e.g., `$AAPL`, `$TSLA`).
2. Plotting the frequency of mentions using visualization libraries like `matplotlib` or `seaborn`.

```python
# Example: Count mentions of $AAPL
stock_symbol = 'AAPL'
mentions = train[train['text'].str.contains(f'\\${stock_symbol}', case=False)]
mentions_count = mentions.shape[0]

# Plot the results
import matplotlib.pyplot as plt
plt.bar(stock_symbol, mentions_count)
plt.show()
```

### Topic Modeling

To identify the key themes or topics being discussed in the tweets, you can use **Latent Dirichlet Allocation (LDA)** or similar techniques. The notebook includes steps for:
1. Preprocessing the tweets (tokenization, removing stopwords).
2. Applying LDA to extract topics from the tweet text.
3. Visualizing the topics using word clouds or bar charts.

```python
from gensim import corpora
from gensim.models.ldamodel import LdaModel

# Preprocess the text data
# Tokenization, removing stopwords, etc.

# Create a dictionary and corpus
dictionary = corpora.Dictionary(preprocessed_texts)
corpus = [dictionary.doc2bow(text) for text in preprocessed_texts]

# Apply LDA model
lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)

# Print the topics
topics = lda_model.print_topics(num_words=5)
for topic in topics:
    print(topic)
```

### Data Scraping (Tweepy)

If you wish to scrape new tweets related to stocks for real-time analysis, use the provided **Data Scraping** notebook:
1. Set up your Twitter API credentials in the `config.py` file.
2. Open the data scraping notebook:
   ```bash
   jupyter notebook Data_Scraping_Tweepy.ipynb
   ```
3. Run the notebook to fetch live tweets. These tweets can then be processed for sentiment analysis, frequency analysis, and topic modeling.

## Contribution

Contributions are welcome! Please follow these steps to contribute to this project:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.
