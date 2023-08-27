from IPython.display import Audio
from bark.generation import (
    generate_text_semantic,
    preload_models,
    generate_coarse,
    codec_decode,
    generate_fine
)
from scipy.io.wavfile import write as write_wav
from bark.api import semantic_to_waveform
from bark import generate_audio, SAMPLE_RATE, preload_models
import numpy as np
import nltk
nltk.download('punkt')

text_prompt = """
    In old times when wishing still helped one, there lived a king whose daughters were all beautiful, but the youngest was so beautiful that the sun itself,
     which has seen so much, was astonished whenever it shone in her face,
    """
def split_sentence(sentence_list):
    potential_split = []
    for i, word in enumerate(sentence_list):
        if ',' in word:
            potential_split.append(i)
    split_no = min(potential_split, key=lambda x: abs(x - len(sentence_list)/2 )) + 1 
    sentence_1 = " ".join(sentence_list[:split_no])
    sentence_2 = " ".join(sentence_list[split_no:])
    return [sentence_1, sentence_2]
    
def validate_sentence(sentence):
    sentence_list = sentence.strip().split(" ")
    if len(sentence_list) > 30:
        print(len(sentence_list))
        print(sentence + 'is over 30 words')
        print("shortening sentence")
        validated_sentences = []
        for sentence in split_sentence(sentence_list):
            validated_sentences.append(validate_sentence(sentence))
        return validated_sentences   
    else:
        return sentence
    
def generate_text(sentence, speaker):
    print(sentence)
    audio_array = generate_audio(sentence, history_prompt=speaker, text_temp=0.7, waveform_temp=0.7)
    silence = np.zeros(int(0.25 * SAMPLE_RATE))
    return [audio_array, silence.copy()]

ex_list = text_prompt.split()
validated_sentences = validate_sentence(text_prompt) 
print(validated_sentences)
             
            
    
# with open('inputs/frog-prince.txt') as file:
#     text_prompt = file.read()

# def short_gen(text_prompt, speaker, file_name):
#     output_path = f'ouputs/{file_name}.wav'
#     base_voice_name = speaker
#     audio_array = generate_audio(text_prompt, history_prompt=speaker, text_temp=0.7, waveform_temp=0.7)
#     Audio(audio_array, rate=SAMPLE_RATE)
#     write_wav(output_path, SAMPLE_RATE, audio_array)

def long_generation(text_prompt, speaker, file_name):    
    sentences = nltk.sent_tokenize(text_prompt, language='english')
    output_path = f'ouputs/{file_name}.wav'
    silence = np.zeros(int(0.25 * SAMPLE_RATE))
    pieces =[]
    for sentence in sentences:
        validated = validate_sentence(sentence)
        if type(validated_sentences) == str:
            pieces += generate_text(validated, speaker)
        else:
            for valid_sentence in validated:
                pieces += generate_text(valid_sentence, speaker)
    final_audio_array = np.concatenate(pieces)
    write_wav(output_path, SAMPLE_RATE, final_audio_array)
    
long_generation(text_prompt, 'lauran-3', 'test-1')
       
            

       




#         
#     pieces += [audio_array, silence.copy()]
#     audio_array = np.concatenate(pieces)
#     
#     Audio(audio_array, rate=SAMPLE_RATE)
