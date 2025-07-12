EXPECTED_MERGE_OUTPUT = [('<BOW>', 'I'), ('<BOW>I', '<EOW>'), ('<BOW>', 'a'), ('<BOW>a', 'm'), ('<BOW>am', '<EOW>'), ('<BOW>', 'u'),
                                                                      ('<BOW>u', 'p'), ('<BOW>up', 's'), ('<BOW>ups', 'k'),
                                                                      ('<BOW>upsk', 'i'), ('<BOW>upski', 'l'), ('<BOW>upskil', 'l'),
                                                                      ('<BOW>upskill', 'i'),
                                                                      ('<BOW>upskilli', 'n'), ('<BOW>upskillin', 'g'),
                                                                      ('<BOW>upskilling', '<EOW>'), ('<BOW>', 'f'), ('<BOW>f', 'r'),
                                                                      ('<BOW>fr', 'e'), ('<BOW>fre', 's'), ('<BOW>fres', 'h'), ('<BOW>fresh', 'e'),
                                                                      ('<BOW>freshe', 'r'), ('<BOW>fresher', '<EOW>'), ('<BOW>', '👳️'),
                                                                      ('<BOW>👳️', '<EOW>')]
EXPECTED_VOCAB = {'<BOW>I': 0,'<BOW>I<EOW>': 1, '<BOW>a': 2, '<BOW>am': 3, '<BOW>am<EOW>': 4, '<BOW>u': 5, '<BOW>up': 6, '<BOW>ups': 7, '<BOW>upsk': 8,
                  '<BOW>upski': 9, '<BOW>upskil': 10, '<BOW>upskill': 11, '<BOW>upskilli': 12, '<BOW>upskillin': 13, '<BOW>upskilling': 14,
                  '<BOW>upskilling<EOW>': 15,'<BOW>f': 16, '<BOW>fr': 17, '<BOW>fre': 18, '<BOW>fres': 19, '<BOW>fresh': 20,
                  '<BOW>freshe': 21, '<BOW>fresher': 22, '<BOW>fresher<EOW>': 23, '<BOW>👳️': 24,  '<BOW>👳️<EOW>': 25, 'a': 26, 'b': 27, 'c': 28,
                  'd': 29, 'e': 30, 'f': 31, 'g': 32, 'h': 33, 'i': 34, 'j': 35, 'k': 36, 'l': 37, 'm': 38, 'n': 39, 'o': 40,
                  'p': 41, 'q': 42, 'r': 43, 's': 44,  't': 45, 'u': 46, 'v': 47, 'w': 48, 'x': 49, 'y': 50, 'z': 51}

