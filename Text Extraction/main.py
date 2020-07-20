from utils import read_pdf, clean_stem_corpus, summary, save_kesimpulan, tfidf_bobot
from nltk.tokenize import sent_tokenize
import time

print('Starting.....')
since = time.time()
PATH = '39775-697-80327-1-10-20180521.pdf'

content = read_pdf(PATH, semua_halaman=False, halaman=1)
corpus = sent_tokenize(content)
print('ukuran corpus', len(corpus))

print('\ncleaning and stemming sentences....')
kalimat_stem = clean_stem_corpus(corpus)
words_weights = tfidf_bobot(kalimat_stem)
print('\nSummarizing.....')
kesimpulan = summary(corpus, words_weights, n_kalimat_summary=5)

save_kesimpulan(kesimpulan, 'kesimpulan.doc')

print(f'\nKesimpulan:\n{kesimpulan}')
print("Waktu: {:.3f} minutes".format((time.time()-since)/60))