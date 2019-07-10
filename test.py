
magazine = 'give me one grand today night'
note = 'give one grand today'
hash_words = {}

# Create the hash tabled with the words on the
# magazine and put the number of occurrence in the value.
for m_word in magazine:
    if hash_words.get(m_word) != None:
        if (hash_words[m_word] > 0):
            hash_words[m_word] += 1
    else:
        hash_words[m_word] = 1

# Check if exist the word in the hash table
for r_word in note:
    if hash_words.get(r_word) is None or hash_words[r_word] == 0:
        print('No')
    else:
        hash_words[r_word] -= 1
print('Yes')