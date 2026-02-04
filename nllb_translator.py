from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import time
import torch
import pandas as pd
import argparse

lang_to_model = {'hindi': 'hin_Deva', 'greek': 'ell_Grek', 'japanese': 'jpn_Jpan', 'spanish': 'spa_Latn'}

def parse_args():
    """
    Parse the arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--noun', type=str, default='house',
                        help='house, car')
    parser.add_argument('--csv-path', type=str, default='None',
                        help='Path to the file')
    parser.add_argument('--language', type=str, default='hindi',
                        help='hindi, greek, japanese, spanish')
    return parser.parse_args()
    
if __name__=='__main__':
    args = parse_args()
    s = time.time()
    noun = args.noun
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B", src_lang=lang_to_model[args.language.lower()])
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B").to('cuda')
    data_path = args.csv_path
    df = pd.read_csv(data_path, engine='python', on_bad_lines='skip')
    print(len(df))

    translations = []
    for idx, row in df.iterrows():
        try:
            print('processed', idx)
            article = row['text'][:50000]
            inputs = tokenizer(article, return_tensors="pt").to('cuda')

            translated_tokens = model.generate(
                **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"), max_length=100
            )
            
            translations.append(tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0])
            
        except:
            print("Exception")
            translations.append('NA')
    e = time.time()
    df['translations_nllb'] = translations
    df.to_csv(data_path, index=False)
    print(e-s)
