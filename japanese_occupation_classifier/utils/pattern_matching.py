from typing import Tuple, Dict, List
import pandas as pd
from .preprocessing import preprocess_text

# Dictionary of occupation patterns for classification
OCCUPATION_PATTERNS: Dict[str, Dict[str, Dict[str, List[str]]]] = {
    "Primary": {
        "agriculture": {
            "farming": [
                "農業", "農家", "農民", "農場", "耕作", "栽培", "田畑", "農作業", "農園", "稲作", 
                "米作", "畑作", "作農", "小作", "自作農", "農耕", "農作", "百姓"
            ],
            "horticulture": [
                "園芸", "果樹", "花卉", "植木", "温室", "ビニールハウス", "野菜作り", "果樹園",
                "花農家", "植木職", "造園", "園芸農家"
            ],
            "livestock": [
                "畜産", "酪農", "養鶏", "養豚", "牧畜", "家畜", "牧場", "養蚕", "酪農家",
                "牧畜業", "畜産農家", "家畜飼育", "養牛", "牧童", "牧夫"
            ],
            "management": [
                "農場主", "農園経営", "農業経営", "農業組合", "農場経営", "農業法人",
                "農業協同組合", "農協職員"
            ]
        },
        "fishing": {
            "marine": [
                "漁業", "漁師", "船員", "水産", "漁労", "漁船", "沿岸漁業", "遠洋漁業", "定置網",
                "漁夫", "船頭", "網元", "漁業従事者", "漁船乗組員", "船長", "水夫"
            ],
            "aquaculture": [
                "養殖", "育成", "栽培漁業", "養魚", "真珠養殖", "のり養殖", "わかめ養殖",
                "魚類養殖", "海藻養殖", "養魚場", "養殖業者"
            ],
            "processing": [
                "水産加工", "魚類加工", "水産物加工", "製塩", "塩干物", "魚介加工",
                "水産加工業", "缶詰製造", "水産食品"
            ]
        },
        "forestry": {
            "logging": [
                "林業", "伐採", "製材", "木材", "森林", "植林", "育林", "山林業",
                "樵", "木こり", "山林労働", "造林", "森林作業"
            ],
            "management": [
                "山林管理", "森林経営", "林業経営", "森林組合", "山林経営",
                "林業事業体", "森林管理者"
            ]
        },
        "mining": {
            "extraction": [
                "鉱業", "採掘", "採鉱", "鉱山", "炭鉱", "金属鉱業", "鉱夫",
                "坑夫", "採炭", "掘削", "鉱山労働"
            ],
            "quarrying": [
                "採石", "砂利採取", "石切", "砕石", "採石場", "石材採取",
                "砂利採取業", "土砂採取"
            ]
        }
    },
    "Secondary": {
        "manufacturing": {
            "general": [
                "製造", "工場", "生産", "製作所", "製造所", "加工場", "作業員",
                "製造業", "工員", "製造工", "工場労働者"
            ],
            "metal": [
                "金属", "製鉄", "鋳造", "金属加工", "製鋼", "鍛造", "板金", "めっき",
                "溶接", "金属工", "製鉄所", "金属製品", "鋳物", "鉄工"
            ],
            "machinery": [
                "機械", "組立", "工作", "機器", "器具", "自動車", "造船", "機械工",
                "旋盤", "精密機械", "機械製造", "自動車製造", "造船所"
            ],
            "textile": [
                "繊維", "紡績", "織物", "染色", "縫製", "織布", "編物", "紡織",
                "織師", "染物", "機織", "裁縫", "洋裁", "和裁"
            ],
            "food": [
                "食品", "加工", "醸造", "製造加工", "製菓", "製パン", "缶詰",
                "食品製造", "菓子製造", "醤油", "味噌", "酒造", "製粉"
            ],
            "chemical": [
                "化学", "製薬", "合成", "化学工業", "石油化学", "プラスチック",
                "化学製品", "医薬品", "化粧品", "塗料", "肥料"
            ]
        },
        "construction": {
            "general": [
                "建設", "建築", "土木", "工事", "建設業", "建築業", "建設作業",
                "建設労働", "土建", "建築現場"
            ],
            "specialized": [
                "大工", "左官", "電気工事", "配管", "建具", "内装", "塗装",
                "とび職", "鳶職", "石工", "タイル", "畳職", "屋根職"
            ],
            "civil": [
                "土木施工", "道路工事", "橋梁", "トンネル", "造園", "土木作業",
                "舗装", "河川工事", "港湾工事"
            ]
        }
    },
    "Tertiary": {
        "office": {
            "general": [
                "会社員", "社員", "従業員", "サラリーマン", "ビジネスマン", "会社勤務",
                "会社役員", "企業職員", "勤め人"
            ],
            "clerical": [
                "事務", "総務", "経理", "秘書", "一般事務", "事務員", "事務職",
                "経理事務", "人事", "庶務", "受付"
            ],
            "management": [
                "管理職", "主任", "課長", "部長", "支配人", "役員", "取締役",
                "経営者", "重役", "執行役員", "社長", "専務", "常務"
            ]
        },
        "commerce": {
            "retail": [
                "販売", "小売", "店員", "商店", "売場", "ショップ", "百貨店",
                "販売員", "売り子", "店舗", "小売店", "商店主"
            ],
            "wholesale": [
                "卸売", "商社", "貿易", "問屋", "仲買", "卸問屋", "卸売業",
                "貿易商", "輸出入", "商事会社"
            ],
            "business": [
                "経営", "事業主", "自営業", "商店主", "取締役", "経営者",
                "企業家", "社長", "事業経営"
            ]
        },
        "professional": {
            "medical": [
                "医師", "医者", "歯科医", "看護師", "薬剤師", "病院", "医療",
                "獣医", "助産師", "保健師", "臨床検査技師"
            ],
            "education": [
                "教師", "教員", "講師", "教授", "学校教師", "塾講師", "教育",
                "先生", "学校職員", "教頭", "校長"
            ],
            "technical": [
                "技術者", "エンジニア", "設計士", "建築士", "技師",
                "システムエンジニア", "プログラマー"
            ]
        }
    },
    "Unemployed": {
        "no_work": {
            "general": [
                "無職", "無業", "なし", "失業", "未就職", "求職中",
                "仕事なし", "職なし"
            ],
            "unable": [
                "休職", "病気", "療養", "障害", "入院", "治療中"
            ]
        },
        "domestic": {
            "homemaker": [
                "主婦", "家事", "専業", "家庭", "主夫", "家事手伝い",
                "家政", "家事従事"
            ],
            "family": [
                "家族従事", "手伝い", "家業", "内職", "家事手伝い",
                "家族労働", "家業手伝い"
            ]
        },
        "student": {
            "general": [
                "学生", "生徒", "児童", "高校生", "中学生"
            ],
            "training": [
                "研修", "修行", "見習", "実習", "養成", "訓練",
                "職業訓練", "技能実習"
            ]
        }
    }
}

def match_occupation(text: str) -> Tuple[str, str, str]:
    """
    Match occupation text to category using predefined patterns.
    
    Args:
        text (str): Input occupation text
        
    Returns:
        Tuple[str, str, str]: (main_category, subcategory, subsubcategory)
    """
    for category, subcategories in OCCUPATION_PATTERNS.items():
        for subcategory, subsubcategories in subcategories.items():
            for subsubcategory, patterns in subsubcategories.items():
                if any(pattern in text for pattern in patterns):
                    return category, subcategory, subsubcategory
    return "Unknown", "unknown", "unknown"

def create_training_data(df: pd.DataFrame, column_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create training data from pattern matching.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column_name (str): Name of the column containing occupation text
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (training_df, unknown_df) where:
            - training_df contains rows with matched categories
            - unknown_df contains rows with unmatched categories
    """
    # Process and clean text
    df['cleaned_occupation'] = df[column_name].apply(preprocess_text)
    
    # Apply pattern matching
    main_categories = []
    subcategories = []
    subsubcategories = []
    
    for text in df['cleaned_occupation']:
        main_cat, subcat, subsubcat = match_occupation(text)
        main_categories.append(main_cat)
        subcategories.append(subcat)
        subsubcategories.append(subsubcat)
    
    # Add results to dataframe
    df['main_category'] = main_categories
    df['subcategory'] = subcategories
    df['subsubcategory'] = subsubcategories
    
    # Filter out Unknown categories for training
    training_df = df[df['main_category'] != "Unknown"].copy()
    unknown_df = df[df['main_category'] == "Unknown"].copy()
    
    return training_df, unknown_df 