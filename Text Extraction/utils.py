import nltk
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import re
import string
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, ArrayDictionary, StopWordRemover
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize, sent_tokenize
import heapq
from tqdm import tqdm
#import PyPDF2

#pip install pdfplumber
import pdfplumber

#stemming (menjadi kata dasar)
stemmerFactory = StemmerFactory()
stemmer = stemmerFactory.create_stemmer()


def read_pdf(PATH, semua_halaman=True, halaman=0):
    """
    membaca file pdf per halamana

    input :
    PATH : lokasi file pdf
    halaman : halaman yg ingin dibuka
    """
    if semua_halaman:
        content = '' # new line
        with pdfplumber.open(PATH) as pdf:
            for pdf_page in pdf.pages:
                single_page_text = pdf_page.extract_text()
                content = content + '\n' + single_page_text
    else:
        with pdfplumber.open(PATH) as pdf:
            page = pdf.pages[halaman]
            content = page.extract_text()
    return content

def cleaning(kalimat):
    """
    membersih notasi dan simbol serta tanda baca yg tidak mewakili kata

    return : 
    kalimat
    """  
    kalimat = kalimat.lower().strip() #huruf kecil dan hapus whitespace
    kalimat = re.sub(r"\d+", "", kalimat) #hapus nomor/angka
    kalimat = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), '', kalimat)  # remove punctuations
    kalimat = re.sub('\s+', ' ', kalimat)  # remove extra whitespace
    return kalimat

def stop_stem_remover(kalimat):
    """
    membersihkan stop word dan melakukan seemming

    input : 
    kalimat : kalimat di dalam corpus

    return
    kalimat
    """
    #buang kata tidak terlalu penting
    #factory = StopWordRemoverFactory()
    stop_factory = StopWordRemoverFactory().get_stop_words()
    add_stop_word = ['dkk', 'et', 'al', 'all'] #tambah manual stopwords
    stop = stop_factory + add_stop_word
    dicts = ArrayDictionary(stop)

    all_stop = StopWordRemover(dicts)
    kalimat = all_stop.remove(kalimat)

    #stemming (menjadi kata dasar)
    stemmerFactory = StemmerFactory()
    stemmer = stemmerFactory.create_stemmer()

    kalimat = stemmer.stem(kalimat)
    return kalimat

def clean_stem_corpus(corpus):
    """
    membersihkan corpus cleaning stop word dan stemming
    dengan menggabungkan def cleaning() dan def stop_stem_remover()

    inputs : 
    corpus : corpus hasil nltk.tokenize.sent_tokenize()

    return :
    kalimat bersih
    """
    #clean
    kalimat_clean = []
    for kal in corpus:
        kalimat_clean.append(cleaning(kal))
    #stemming
    kalimat_stem = []
    for kal in tqdm(kalimat_clean):
        kalimat_stem.append(stop_stem_remover(kal))
    return kalimat_stem

def tfidf_bobot(kalimat_stem):
    """
    menghitung nilai tfidf dari corpus hasil cleaning dan stemming
    input : kalimat yg telah dibersihkan

    return : dictionary kata dan nilai tfidf
    """
    #hitung tfidf
    tf_idf_vec = TfidfVectorizer()
    tf_idf_metric = tf_idf_vec.fit_transform(kalimat_stem)

    words = tf_idf_vec.get_feature_names() #kosa kata
    tfidf_val = tf_idf_metric.toarray() 

    avarage_tfidf = tfidf_val.sum(0) / (tfidf_val != 0).sum(0) #rata" untuk tfidf per kata dari semua kalimat
    words_weights = dict(zip(words, avarage_tfidf)) #dictionari kata dan nilai tfidf ny
    return words_weights

def summary(corpus, words_weights, n_kalimat_summary=3):
    """
    membuat kesimpulan dengan menjumlahkan tfidf setiap kata dalam kalimat

    input:
    corpus : kalimat-kalimat hasil nltk.sent_tokenize (pisah per kalimat)
    words_weights : bobot tfidf per kata
    n_kalimat_summary : jumlah minimal kalimat kesimpulan

    return:
    kesimpulan
    """

    sentence_rank={}
    for sent in tqdm(corpus): #loop setiap kalimat dalam corpus
        for kata in word_tokenize(stemmer.stem(" ".join(word_tokenize(sent.lower())))):
            #loop setiap kata (hasil stemming, karena kalimat dalam corpus merupakan kata asli sedangkan 
            # key dari words_weights-tfidf hasil kata stem) pada setiap kalimat
            if kata in words_weights.keys():
                #jika ditemukan kata dalam words_weights 
                if sent in sentence_rank.keys():
                    sentence_rank[sent] += words_weights[kata]
                    #jumlahkan tfidf
                else:
                    sentence_rank[sent] = words_weights[kata]
                    #jika kalimat belum ada dalam sentence_rank tambahkan
            else:
                #kalau tidak ditemukan kata kunci lewat saja
                continue

    summary_sentences = heapq.nlargest(n_kalimat_summary, sentence_rank, key=sentence_rank.get)

    summary = ' '.join(summary_sentences)
    return summary

def save_kesimpulan(kesimpulan, path):
    """
    menyimpan hasil kesimpulan dalam bentuk file

    input:
    kesimpulan : hasil string summary
    path : nama dan lokasi file untuk disimpan beserta ekstensi file
           contoh= summary.txt, summary.doc
    """
    with open(path, 'w') as output:
        output.write(kesimpulan)