def test_encode(tokenizer):
    tokenizer.encode(
        [
            ["TÚL", "MI", "this_is_unkown_word"],
            [
                "TÚL",
                "MI",
                "PAB",
                "AB",
                "NI",
                "GIŠ",
                "ZÍ",
                "TI",
                "GEŠTIN",
                "IM",
                "GAG",
                "GAN",
                "SU",
                "UR",
                "ANŠE",
                "KIN",
                "MAR",
            ],
        ]
    )
