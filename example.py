from medkeywordsfinder import medKeywordsFinder

if __name__ == "__main__":
    finder = medKeywordsFinder()
    finder.load()
    topK = 10
    question_list = [
        "What are the side effects of escitalopram?",
        "Which genes are associated with breast cancer?",
        "How does metformin help in type 2 diabetes treatment?",
        "What are the approved COVID-19 vaccines?",
        "Which proteins are involved in Alzheimer's disease?",
        "How does CRISPR-Cas9 work in gene editing?",
        "What is the mechanism of action of ibuprofen?",
        "Which antibiotics are effective against MRSA?",
        "How is Parkinson’s disease diagnosed?",
        "What biomarkers indicate early-stage lung cancer?"]
    

    for question in question_list:
        kws = ', '.join(finder.search(question,topK=topK))
        print(f"question: {question}\ntop 10 keywords: {kws}\n")