from transformers import AutoTokenizer
from config import BASE_MODEL

if __name__ == '__main__':
    text = 'I was charged twice for my subscription.'
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    encoded = tokenizer(text, truncation=True, max_length=256)
    print('Text:', text)
    print('Token strings:', tokenizer.convert_ids_to_tokens(encoded['input_ids']))
    print('Token IDs:', encoded['input_ids'])
    print('Attention mask:', encoded['attention_mask'])
    print('Token IDs are vocabulary identifiers; the model turns them into vectors.')
