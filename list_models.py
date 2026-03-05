import ollama
with open('models.txt', 'w', encoding='utf-8') as f:
    try:
        models = ollama.list()['models']
        for m in models:
            f.write(m['model'] + '\n')
    except Exception as e:
        f.write(str(e))
