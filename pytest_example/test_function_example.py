# content of test_sample.py
def inc(x):
    return x + 1


def test_answer():
    assert inc(3) == 4

def test_answer2():
    assert inc(3) == 4

def test_answer3():
    assert inc(3) == 4

def test_answer4():
    assert inc(3) == 5  # This test will fail