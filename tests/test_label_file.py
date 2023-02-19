def test_label_file(label_file):
    correct = ["ḫar", "na", "a", "ú", "i", " ", "QA", "TAM", " ", "pa", "ra", "a"]
    assert correct == label_file.readings
