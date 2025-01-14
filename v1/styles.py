MAIN_STYLE = """
    <style>
        .title {
            text-align: center;
            background: linear-gradient(45deg, #C9082A, #17408B);
            color: white;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 30px;
            font-size: 48px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        .model-comparison {
            display: flex;
            justify-content: space-between;
            margin: 30px 0;
            gap: 24px;
        }
        .model-card {
            flex: 1;
            background: white;
            padding: 24px;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            text-align: center;
            transition: transform 0.2s, box-shadow 0.2s;
            border: 2px solid;
        }
        .model-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 25px rgba(0,0,0,0.12);
        }
        .rf-card { border-color: #17408B; }
        .dnn-card { border-color: #C9082A; }
        .actual-card { border-color: #2E8B57; }
        .player-card {
            background: white;
            padding: 30px;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            margin: 30px 0;
            border-left: 5px solid #17408B;
        }
    </style>
"""