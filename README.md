# Symptom-Based Disease Classification Using BioBERT

## Description
This project develops a Natural Language Processing (NLP) framework for classifying diseases based on symptom descriptions using BioBERT, a transformer model pre-trained on biomedical corpora. Leveraging the UCI ML Drug Review Dataset, the model processes patient-reported symptom data to assist in preliminary medical diagnostics. The current implementation achieves 40% accuracy, with a projected 70-85% upon full optimization. The project addresses challenges like dataset imbalance and symptom ambiguity, aiming to provide a scalable solution for real-time medical applications.

## Features
- **BioBERT-Powered Classification**: Utilizes BioBERT’s contextual embeddings, fine-tuned for symptom-based disease classification.
- **Robust Preprocessing**: Includes label encoding, tokenization, stratified splitting, and tensor dataset creation for efficient training.
- **Handles Real-World Data**: Processes patient-authored text from drug reviews, capturing real-world symptom variability.
- **Scalable Architecture**: Designed for multi-class classification with potential for web-based deployment.
- **Performance Evaluation**: Assessed using accuracy and F1-score to address dataset imbalance.

## Installation
```bash
git clone https://github.com/your-username/biobert-disease-classification.git
cd biobert-disease-classification
pip install -r requirements.txt
```

### Prerequisites
- Python 3.8+
- PyTorch
- Hugging Face Transformers
- Scikit-learn, Pandas, Matplotlib
- GPU (recommended for faster training)

## Usage
1. **Prepare the Dataset**:
   - Download the UCI ML Drug Review Dataset from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018).
   - Place it in the `data/` directory and preprocess it:
     ```bash
     python preprocess.py --input data/drug_reviews.csv --output data/processed_dataset.csv
     ```

2. **Train the Model**:
   ```bash
   python train.py --dataset data/processed_dataset.csv --batch-size 128 --epochs 5
   ```

3. **Evaluate the Model**:
   ```bash
   python evaluate.py --model models/biobert_classifier.pth --test data/processed_dataset.csv
   ```

## Dataset
The UCI ML Drug Review Dataset contains patient reviews with symptom descriptions, conditions, and medication effectiveness. Key details:
- **Records**: Patient-authored text data.
- **Features**: Symptom descriptions and disease labels.
- **Preprocessing**:
  - **Label Encoding**: Maps disease labels to integers.
  - **Tokenization**: Uses BERT tokenizer with max length=128, padding, and truncation.
  - **Stratified Splitting**: 90% training, 5% validation, 5% test.
  - **DataLoader**: Batched (size=128) and shuffled for efficient training.

## Methodology
The BioBERT-based classifier operates as follows:
1. **Preprocessing**:
   - Symptom descriptions are tokenized using the BERT tokenizer.
   - Data is split into training, validation, and test sets with stratified sampling.
2. **Model Architecture**:
   - **Pre-trained BioBERT**: Generates contextual embeddings for symptom text.
   - **Fully Connected Layer**: Maps embeddings to disease classes.
   - **Softmax Activation**: Outputs probability distributions over classes.
   - **Optimization**: Uses AdamW optimizer with cross-entropy loss.
3. **Training**:
   - Fine-tunes BioBERT over multiple epochs with a learning rate scheduler.
   - Processes data in mini-batches for efficiency.
4. **Inference**:
   - Evaluates test data using accuracy and F1-score.

### Pseudocode
```python
# Load and preprocess dataset
dataset = load_dataset('data/drug_reviews.csv')
encoded_labels = encode_labels(dataset['labels'])
tokenized_data = bert_tokenize(dataset['symptoms'], max_length=128)
train_data, val_data, test_data = stratified_split(tokenized_data, ratios=[0.9, 0.05, 0.05])

# Initialize BioBERT model
model = BioBERTForClassification(pretrained='bioBERT', num_classes=len(unique_labels))
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
for epoch in range(epochs):
    for batch in DataLoader(train_data, batch_size=128, shuffle=True):
        outputs = model(batch['input_ids'], batch['attention_mask'])
        loss = compute_loss(outputs, batch['labels'])
        optimizer.step()

# Evaluation
predictions = model.predict(test_data)
accuracy, f1 = evaluate(predictions, test_data['labels'])
```

## Results
The BioBERT model’s performance is summarized below:
- **Current Accuracy**: 40%
- **Projected Accuracy**: 70-85% (with full optimization)
- **Evaluation Metrics**: Accuracy and macro-averaged F1-score to account for class imbalance.

### Performance Comparison
| Model       | Current Accuracy | Projected Accuracy |
|-------------|------------------|--------------------|
| BioBERT     | 40%              | 70-85%            |
| MiniBERT    | 15-20%           | 55-60%            |
| LSTM        | 15-20%           | 40-45%            |

BioBERT outperforms baseline models (MiniBERT, LSTM) for common diseases but struggles with rare diseases due to dataset imbalance and symptom ambiguity.

## Challenges
- **Dataset Imbalance**: Underrepresented diseases lead to biased predictions.
- **Computational Cost**: Fine-tuning BioBERT requires significant GPU resources.
- **Symptom Ambiguity**: Overlapping symptoms (e.g., fever, fatigue) hinder accurate classification.
- **Non-Standardized Text**: Patient-authored reviews lack consistent terminology.

## Future Work
- **Data Augmentation**: Use back-translation or GAN-based methods to generate synthetic samples for rare diseases.
- **Hybrid Models**: Combine BioBERT with LSTM or CNN for better sequential dependency capture.
- **Advanced Architectures**: Explore RoBERTa, XLNet, or ClinicalBERT for improved performance.
- **Medical Knowledge Integration**: Incorporate ontologies (e.g., UMLS, SNOMED CT) for enriched context.
- **Web Deployment**: Develop a web-based application for real-time symptom-based diagnostics.

## References
1. C. Dreisbach et al., “A systematic review of natural language processing and text mining of symptoms from electronic patient-authored text data,” *Int. J. Med. Inform.*, vol. 125, pp. 37–46, 2019.
2. A. Johnson et al., “Natural language processing of symptoms documented in free-text narratives of electronic health records: A systematic review,” *J. Am. Med. Inform. Assoc.*, vol. 26, no. 2, pp. 164–174, 2019.
3. B. Smith et al., “Natural language processing to extract symptoms of severe mental illness from clinical text: A comparative evaluation,” *Psychiatr. Serv.*, vol. 68, no. 5, pp. 456–462, 2017.
4. C. Lee et al., “Extraction of disease symptoms from free text using natural language processing,” *J. Health Inform. Res.*, vol. 7, no. 3, pp. 321–335, 2023.
5. S. Kumar et al., “An evaluation of clinical natural language processing systems to extract adverse event symptoms,” *J. Biomed. Inform.*, vol. 115, p. 103694, 2021.
6. Y. Wang et al., “Natural language processing of nursing notes: A systematic review,” *Nurs. Res.*, vol. 70, no. 4, pp. 287–297, 2021.
7. T. Chen et al., “Modern clinical text mining: A guide and review,” *Annu. Rev. Biomed. Data Sci.*, vol. 3, pp. 165–187, 2020.
8. R. Jones et al., “Applying natural language processing and machine learning techniques to patient-authored text: A systematic review,” *J. Med. Internet Res.*, vol. 23, no. 5, p. e25681, 2021.
9. L. Brown et al., “What do patients say about their disease symptoms? Deep multilabel text classification with human-in-the-loop curation for automatic labeling of patient self-reports of problems,” *J. Am. Med. Inform. Assoc.*, vol. 30, no. 6, pp. 1098–1106, 2023.
10. P. Adams et al., “Adaptation of IDPT system based on patient-authored text data using NLP,” *Artif. Intell. Med.*, vol. 115, p. 102057, 2021.
11. M. Patel et al., “Extracting medical information from clinical text with NLP,” *Health Inform. J.*, vol. 29, no. 2, pp. 1–15, 2023.
12. H. Li et al., “Clinical NLP: State-of-the-art natural language processing to extract information from clinical notes,” *J. Biomed. Inform.*, vol. 127, p. 104013, 2022.
13. Q. Zhang et al., “Patient symptom extractor from history of illness notes (PSEHI): A deep learning approach,” *Comput. Biol. Med.*, vol. 141, p. 105112, 2022.
14. J. Li, “UCI ML Drug Review Dataset,” *Kaggle*, 2018. [Online]. Available: https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018
15. J. Lee et al., “BioBERT: A pre-trained biomedical language representation model for biomedical text mining,” *Bioinformatics*, vol. 36, no. 4, pp. 1234–1240, 2020.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or collaboration, reach out to:
- **Kush Revankar**
- **Email**: kushrevankar24@gmail.com
- **Institution**: MIT World Peace University, Pune, India
