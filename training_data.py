"""
Training Data Generator for AI/Human Text Detection
Generates synthetic training data with distinguishing features
"""

# Sample AI-generated text characteristics:
# - More formal and structured
# - Uses varied vocabulary and sentence structures
# - Often includes transitional phrases
ai_generated_samples = [
    # English samples
    "Artificial intelligence has revolutionized numerous industries, transforming the way we approach complex problem-solving. Furthermore, machine learning algorithms have demonstrated remarkable capabilities in pattern recognition and predictive analytics. Consequently, organizations worldwide are increasingly adopting these technologies to enhance operational efficiency.",
    "The implementation of neural networks represents a significant advancement in computational capabilities. These sophisticated systems process information through interconnected layers, mimicking biological neural processes. Moreover, deep learning architectures have achieved unprecedented accuracy in various domains, including computer vision and natural language processing.",
    "Climate change poses substantial challenges for global ecosystems. Scientists have documented rising temperatures and shifting weather patterns across continents. Additionally, the impacts extend beyond environmental concerns, affecting economic stability and social structures. Therefore, comprehensive strategies are essential for mitigation and adaptation.",
    "The evolution of digital communication has fundamentally altered interpersonal relationships. Social media platforms facilitate instantaneous connections across geographical boundaries. However, concerns regarding privacy and misinformation have emerged as critical issues. Consequently, users must navigate these technologies with increased awareness and discretion.",
    "Quantum computing represents a paradigm shift in information processing. Unlike classical computers that utilize binary states, quantum systems leverage superposition and entanglement. This enables exponentially faster computation for specific problem classes. Nevertheless, significant technical challenges remain before widespread practical implementation becomes feasible.",
    "The education sector has undergone substantial transformation through technology integration. Online learning platforms provide unprecedented access to educational resources. Furthermore, adaptive learning systems personalize instruction based on individual student needs. However, ensuring equitable access remains a persistent challenge across diverse socioeconomic contexts.",
    "Renewable energy sources have gained prominence as sustainable alternatives to fossil fuels. Solar and wind technologies have experienced dramatic cost reductions, improving economic viability. Moreover, energy storage solutions address intermittency challenges. Consequently, the transition toward clean energy accelerates globally, supporting environmental objectives.",
    "Biotechnology innovations have expanded the frontiers of medical treatment. Gene editing technologies like CRISPR enable precise modifications to genetic sequences. Additionally, personalized medicine approaches tailor treatments to individual genetic profiles. These advancements offer promising solutions for previously intractable conditions.",
    "The Internet of Things connects billions of devices, generating vast data streams. This interconnected network enables smart homes, cities, and industrial systems. Furthermore, data analytics extract valuable insights from these information flows. However, security vulnerabilities and privacy concerns require careful attention and robust solutions.",
    "Space exploration continues to captivate human imagination and scientific inquiry. Recent missions have revealed fascinating details about distant planets and celestial bodies. Moreover, private companies have entered the aerospace sector, accelerating innovation and reducing costs. Consequently, ambitious projects like Mars colonization appear increasingly achievable.",
    "The financial technology sector disrupts traditional banking and payment systems. Digital currencies and blockchain technology offer decentralized alternatives. Additionally, mobile payment solutions provide convenient transaction methods. However, regulatory frameworks struggle to keep pace with rapid technological evolution.",
    "Cybersecurity threats have escalated in sophistication and frequency. Malicious actors employ advanced techniques to compromise systems and data. Furthermore, the increasing digitization of critical infrastructure amplifies potential vulnerabilities. Therefore, organizations must implement comprehensive security strategies and continuous monitoring.",
    "Agricultural technology enhances food production efficiency and sustainability. Precision farming techniques optimize resource utilization through data-driven decision-making. Moreover, vertical farming and hydroponics enable cultivation in limited spaces. These innovations address growing food security concerns amid population expansion.",
    "The entertainment industry has been transformed by streaming platforms and digital distribution. Content creators reach global audiences without traditional gatekeepers. Additionally, recommendation algorithms personalize viewing experiences. However, concerns about content moderation and creator compensation persist.",
    "Urban planning increasingly incorporates smart city concepts and sustainable design principles. Integrated transportation systems reduce congestion and emissions. Furthermore, green spaces and efficient buildings improve quality of life. These comprehensive approaches address complex challenges of urbanization and environmental stewardship.",
    # Chinese AI samples
    "人工智慧技術在當代社會中扮演著日益重要的角色。隨著深度學習算法的不斷發展，機器學習系統已經在多個領域取得了顯著成就。此外，自然語言處理技術的進步使得機器能夠更好地理解人類語言。因此，人工智慧應用正在快速滲透到各個產業之中。",
    "氣候變遷對全球生態系統構成重大挑戰。科學研究表明，溫室氣體排放持續增加導致全球氣溫上升。此外，極端氣候事件的頻率和強度都在不斷增加。因此，國際社會必須採取積極行動，推動可持續發展策略的實施。",
    "數位轉型已成為企業發展的必然趨勢。雲端運算技術為企業提供了靈活的資源配置方案。同時，大數據分析能力使得企業能夠更好地洞察市場需求。綜上所述，數位化創新已成為提升競爭力的關鍵因素。",
    "教育體系正經歷深刻的變革過程。線上學習平台打破了傳統教育的時空限制。此外，個性化學習系統能夠根據學生的具體需求調整教學內容。然而，如何確保教育公平性仍然是一個需要持續關注的重要議題。",
    "區塊鏈技術為金融領域帶來了革命性的變革。分散式賬本系統提供了更高的透明度和安全性。同時，智能合約技術自動化了許多複雜的交易流程。因此，區塊鏈應用正在逐步改變傳統金融服務模式。",
    "量子計算代表了計算技術的重大突破。與傳統計算機不同，量子系統利用量子疊加和量子糾纏等特性進行運算。這種全新的計算範式為解決複雜問題提供了可能性。然而，量子計算的實際應用仍然面臨諸多技術挑戰。",
    "物聯網技術正在構建萬物互聯的智慧世界。感測器網路收集大量實時數據，為決策提供支持。此外，邊緣計算技術提升了數據處理的效率。因此，物聯網應用正在深刻改變人們的生活方式。",
    "生物科技的發展為醫療領域開闢了新的可能性。基因編輯技術使得精準治療成為現實。同時，再生醫學研究為組織修復提供了創新方案。這些技術進步為解決重大疾病帶來了新的希望。",
    "電子商務平台已經徹底改變了零售業態。透過大數據分析和人工智慧技術，企業能夠提供個性化的購物體驗。此外，供應鏈優化系統提升了配送效率。因此，線上購物已成為消費者的主要選擇之一。",
    "虛擬實境技術為娛樂產業帶來了革命性的變革。沉浸式體驗使得用戶能夠進入虛擬世界進行互動。同時，增強實境應用將數位內容疊加到現實環境中。這些創新技術正在重新定義娛樂體驗的邊界。",
    "資訊安全已成為數位時代的核心議題。網路攻擊手段日益複雜，對企業和個人都構成威脅。此外，隱私保護法規的完善推動了安全技術的發展。因此，建立完善的安全防護體系至關重要。",
    "環境保護與經濟發展之間需要達成平衡。綠色科技的應用為可持續發展提供了解決方案。同時，循環經濟模式減少了資源浪費。這些措施共同推動了環保目標的實現。",
]

# Human-written text characteristics:
# - More personal and conversational
# - Natural flow with occasional imperfections
# - Varied sentence lengths and informal expressions
human_written_samples = [
    "I've been thinking about AI lately and honestly it's pretty wild how far it's come. Like, just a few years ago this stuff seemed impossible but now it's everywhere. My phone can recognize my face, my car can kinda drive itself (though I don't trust it completely lol), and these chatbots are getting scary good at pretending to be human.",
    "So I tried making sourdough bread last weekend. Total disaster! The dough didn't rise properly and I think I might have killed my starter. My friend Sarah makes it look so easy but apparently there's some secret technique I'm missing. Maybe I'll just stick to buying bread from the bakery down the street.",
    "Climate change is really scary when you think about it. I mean we're seeing crazy weather everywhere - floods, fires, droughts. And it feels like nobody's doing enough about it? Idk, sometimes I feel pretty helpless about the whole situation. But I guess small things like recycling and using less plastic help a bit.",
    "Just got back from an amazing vacation in Japan! The food was incredible - I must have eaten sushi like every day. Tokyo was super crowded but also really clean and organized. Everyone was so polite too. Already planning my next trip back because two weeks definitely wasn't enough time to see everything.",
    "My little sister is obsessed with quantum physics now. She keeps trying to explain superposition to me but honestly most of it goes over my head. Something about particles being in multiple states at once? It sounds like magic to me but she insists it's real science. Kids these days are way smarter than I was at her age.",
    "Online classes are okay I guess but I really miss being on campus. It's hard to stay motivated when you're just staring at a screen all day. Plus my internet connection keeps cutting out during important lectures which is super annoying. Can't wait for things to go back to normal.",
    "Been trying to reduce my electricity bill by using more solar power. Installed some panels on the roof last month and it's actually working pretty well! The initial cost was pretty steep but I figure it'll pay for itself eventually. Plus it feels good to do something positive for the environment.",
    "Went to the doctor yesterday for a checkup. Everything looks good thankfully but she mentioned some new genetic testing they can do now. It's supposed to predict your risk for certain diseases or something. Not sure how I feel about knowing all that information about myself honestly. Would you want to know?",
    "My house is slowly turning into a smart home and I'm not sure if I love it or find it creepy. The thermostat learns my schedule, the lights turn on automatically, and my fridge can apparently order groceries. It's convenient but also feels like I'm living in a sci-fi movie sometimes.",
    "There's this documentary about Mars missions that I watched last night. Pretty fascinating stuff! The idea that humans might actually live on another planet in my lifetime is crazy. Though honestly I think I'll stay here on Earth - Mars looks cold and dusty and there's no pizza delivery.",
    "Trying to get better at managing my money so I downloaded this budgeting app. It's actually kind of eye-opening to see where all my money goes. Apparently I spend way too much on coffee and takeout food. Oops. Time to start cooking more at home I guess.",
    "Got a weird email today that was definitely a phishing attempt. They pretended to be from my bank but the grammar was all wrong and the link looked sketchy. It's concerning how convincing these scams are getting though. My mom almost fell for one last month.",
    "My neighbor started a little vegetable garden in his backyard and it's making me want to try it too. He's growing tomatoes, peppers, cucumbers - all sorts of stuff. Says it's really satisfying to eat food you grew yourself. Maybe I'll start with something easy like herbs.",
    "Binge-watched an entire series on Netflix this weekend. No regrets! Well maybe a few regrets because I should have been doing homework but whatever. The show was really good - kept me guessing until the end. Already looking for something new to watch.",
    "The city wants to redesign our downtown area and they're asking for public input. I went to one of the community meetings and there were some interesting ideas. More bike lanes, maybe a park where that old parking lot is, better public transit. Hope they actually listen to what people want.",
    # Chinese human samples
    "最近在想AI這個東西真的很神奇欸。感覺前幾年還覺得很遙遠的技術，現在突然就在生活中到處都是了。手機可以人臉辨識，還有那些聊天機器人越來越像真人了，有時候還真的分不出來lol。不知道未來會變成怎樣。",
    "上週末試著做麵包結果超級失敗！麵團完全發不起來，我覺得我可能把酵母弄死了哈哈。看我朋友做都超簡單的，但我就是做不好。算了，還是去麵包店買比較實在，而且現在麵包店選擇也很多。",
    "氣候變遷這件事想想真的很可怕。到處都在發生極端氣候，水災、火災、乾旱什麼的。感覺政府做的還是不夠多？有時候覺得個人能做的真的很有限，不過還是盡量做環保啦，能做多少算多少。",
    "剛從日本旅遊回來！真的超好玩的，食物超好吃，我每天都在吃壽司和拉麵。東京人超多但是環境很乾淨，而且日本人都好有禮貌喔。已經在計劃下次什麼時候再去了，兩個禮拜根本不夠玩。",
    "我妹最近超迷量子物理的，一直跟我講什麼量子疊加態之類的，但我真的聽不太懂。好像是說粒子可以同時存在多個狀態？聽起來好像魔法一樣，但她說這是真的科學。現在的小孩真的比我那時候聰明太多了。",
    "線上課程上起來還可以啦，但真的很想念在學校上課的感覺。整天盯著螢幕真的很難專心，而且網路還常常斷線，考試的時候超緊張的。希望疫情趕快過去，可以回學校正常上課。",
    "最近在試著省電費，裝了太陽能板在屋頂上。雖然一開始花了不少錢，但長期來看應該划算吧。而且感覺對環境也比較好，做一點自己能做的事情。鄰居看到也想裝了，哈哈。",
    "昨天去醫院做健康檢查，還好一切正常。醫生說現在有基因檢測可以預測疾病風險，聽起來很厲害但我不確定想不想知道那麼多。萬一測出來有問題不是更擔心嗎？你會想做這種檢測嗎？",
    "我家現在越來越smart了，不知道該說方便還是有點詭異。冷氣會自己調溫度，燈會自動開關，冰箱還能自己訂貨。雖然很方便但有時候覺得好像被監控一樣，有點毛毛的。",
    "昨天看了一部關於火星探索的紀錄片，超酷的！想到人類可能在我有生之年就能住在火星上，真的很不可思議。不過老實說我還是比較想待在地球上啦，火星看起來又冷又荒涼，而且沒有珍奶。",
    "開始用記帳app管理開支，發現自己花太多錢在飲料和外食上了。每天一杯手搖飲料加上午餐晚餐都外食，難怪存不了錢。看來要開始學著自己煮飯了，但我廚藝真的很差哈哈。",
    "今天收到一封超假的釣魚信件，假裝是銀行寄來的但文法超爛，一看就知道是詐騙。不過聽說現在詐騙手法越來越高明，我媽上次差點就被騙了。大家真的要小心一點。",
    "鄰居在後院種菜，看起來好像很不錯。他種了番茄、辣椒、小黃瓜什麼的，說吃自己種的菜很有成就感。我也想試試看，但不知道從哪裡開始，可能先種個簡單的香草類好了。",
    "這週末把一整部Netflix影集看完了，完全沒後悔！雖然該寫的作業都還沒動，但那部劇真的太好看了，一直讓人想繼續看下去。現在又在找下一部要看的影集了。",
    "去逛夜市買了一堆小吃，鹽酥雞、雞排、珍珠奶茶通通來一份。結果吃太撐走不動了哈哈。不過夜市的氣氛真的很棒，很熱鬧很有台灣味，朋友從國外回來都說超想念這個。",
    "最近迷上烘焙，試著做了幾次蛋糕但都沒很成功。不是烤焦就是沒熟，溫度真的好難控制。看YouTube教學感覺很簡單，實際做起來才發現好多眉角，還是需要多練習啦。",
    "公司最近在推行遠距工作，感覺蠻不錯的可以省通勤時間。但在家工作有時候會偷懶，專注力沒那麼好。而且少了跟同事面對面聊天的機會，有點寂寞欸。不過整體來說還是喜歡這種彈性。",
    "昨天去看演唱會超嗨的！歌手唱現場比錄音室版本更有感覺，氣氛整個炸裂。旁邊的人一起合唱的時候真的會起雞皮疙瘩，那種感動很難形容。下次有機會一定還要再去。",
    "朋友推薦我一家新開的火鍋店，說很好吃。結果去排了一個小時的隊才吃到，但真的值得等！湯頭超讚料也很新鮮，價格也還算合理。已經跟家人約好下禮拜再去吃一次了。",
    "最近在學吉他，手指按弦按到超痛的。剛開始連簡單的和弦都按不好，轉換也很慢。不過慢慢有進步的感覺還不錯，希望之後可以彈出完整的歌。老師說要多練習才會熟練。",
    "週末去爬山呼吸新鮮空氣，風景超美的而且空氣很好。雖然爬到一半有點累但到山頂看到view的時候覺得很值得。下山之後腿超酸，但心情很放鬆，下次還想再去別的山。",
    "在網路上看到一個好笑的梗圖，笑到不行。傳給朋友他們也都覺得超好笑，一直在群組裡討論。網路文化真的變化很快，每天都有新的梗出現，有時候跟不上就會落伍了哈哈。",
]

def get_training_data():
    """
    Returns training data for the AI detector model
    Returns: texts (list), labels (list) where 1=AI, 0=Human
    """
    texts = ai_generated_samples + human_written_samples
    labels = [1] * len(ai_generated_samples) + [0] * len(human_written_samples)
    return texts, labels

if __name__ == "__main__":
    texts, labels = get_training_data()
    print(f"Total samples: {len(texts)}")
    print(f"AI samples: {sum(labels)}")
    print(f"Human samples: {len(labels) - sum(labels)}")
