from flask import Flask, render_template, Response, jsonify, request
import cv2
import speech_recognition as sr
import numpy as np
from tensorflow.keras.models import load_model as load_keras_model

app = Flask(__name__)
r = sr.Recognizer()

# Keras yüz duygu tanıma modelini yükle
facial_emotion_model = load_keras_model("facialemotionmodel.h5")
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

camera = cv2.VideoCapture(0)

def gen_frames():  # Video stream generator function
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = roi_gray / 255.0
                roi_gray = np.expand_dims(roi_gray, axis=0)
                roi_gray = np.expand_dims(roi_gray, axis=-1)

                prediction = facial_emotion_model.predict(roi_gray)
                maxindex = int(np.argmax(prediction))
                emotion = labels[maxindex]

                cv2.rectangle(frame, (x, y), (x+w, h+y), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Doğrudan kodda duygu ifadeleri
duygu_ifadeleri =  {
        "happy": [
            "mutlu", "neşeli", "hoşnut", "gülümseyen", "sefa dolu", "umutlu", "kahkahalı", "coşkulu",
            "şenlikli", "ferahlık", "güzel", "heyecan", "sevinçli", "keyifli", "havalı", "inanılmaz",
            "muhteşem", "şahane", "çok hoş", "gülmek", "harika", "mükemmel", "sevinç", "coşku",
            "bayram havası", "kutlama", "parti", "zafer", "şanslı", "ne kadar güzel", "aferin",
            "harikasın", "müthiş", "süper", "olağanüstü", "neşe dolu", "pırıl pırıl", "umut verici",
            "dost canlısı", "ışıl ışıl", "moralim çok yüksek", "iyi ki varım", "bu harika bir gün",
            "hayat çok güzel", "dünya çok güzel", "yaşamak çok güzel", "like", "happy", "joyful", "delighted",
            "cheerful"
        ],
        "sad": [
            "üzgün", "hüzünlü", "kederli", "morali bozuk", "mahzun", "gamlı", "kötü", "cansıkıcı",
            "melankolik", "hay aksi", "of", "ne yazık", "kahretsin", "hasta", "mutsuz", "gözyaşı",
            "kırgın", "yalnız", "çaresiz", "umutsuz", "depresif", "karamsar", "üzücü", "dertli",
            "gözlerim dolu", "canım sıkkın", "kalbim kırık", "yıkıldım", "içim karardı", "güçsüz",
            "yorgun", "bitkin", "hayal kırıklığı", "düş kırıklığı", "neden ben", "acı çekiyorum",
            "zayıf", "ağlamak", "içim parçalanıyor", "yeter artık", "dayanamıyorum", "çok kötü hissediyorum", "sad",
            "unhappy", "down", "depressed", "sorrowful"
        ],
        "angry": [
            "kızgın", "öfkeli", "sinirli","hiddetli", "öfke içinde", "hırslı", "telaşlı", "sakin değil",
            "galeyana gelmiş", "çılgına dönmüş", "yapma", "öfkeliyim", "çıldırmak üzereyim", "sinir bozucu",
            "bağırmak", "delirmiş", "kontrolsüz", "patlamak üzere", "hiddetlenmiş", "sert", "inatçı",
            "saldırgan", "nefret", "hayal kırıklığı", "tahammülsüz", "hırçın", "asabi", "öfkemden kuduruyorum",
            "kafayı yemek üzereyim", "dayanamıyorum", "artık yeter", "lanet olsun", "herkes çıksın",
            "çok sinirlendim", "kendimi kaybediyorum", "patlayacağım", "çık dışarı", "lanet olası",
            "nefret ediyorum", "sinirlerim bozuldu", "don't", "angry", "furious", "enraged", "irritated", "mad"
        ],
        "surprise": [
            "şaşkın", "şaşırmış", "hayret etmiş", "şaşırtıcı", "şaşkına dönmüş", "afallamış", "şok olmuş",
            "hayrete düşmüş", "şaşkın bakışlarla", "şaşırmış bir şekilde", "hayret içinde", "şaşkınlıkla",
            "vay canına", "ne garip", "aman Allah'ım", "hıı", "nasıl yani", "şaşırdım kaldım",
            "gerçekten mi", "inanamıyorum", "beklenmedik", "şoke edici", "hayal bile edemezdim", "inanılmaz",
            "akıl almaz", "hayret verici", "çok ilginç", "olağanüstü", "şaşırmış haldeyim", "ağzım açık kaldı",
            "bu nasıl olur", "mümkün değil", "gerçek mi bu", "inanılmaz bir şey", "hayretler içindeyim",
            "şaşkınım", "şoktayım", "beklemiyordum", "surprised", "astonished", "amazed", "shocked", "astounded"
        ],
        "fear": [
            "korkmuş", "korku içinde", "endişeli", "ürkmüş", "dehşete kapılmış", "dehşet içinde",
            "endişeyle dolmuş", "korkuya kapılmış", "korku dolu", "titrek", "şaşırmış", "panik içinde",
            "ter içinde", "çaresiz", "dehşetli", "ürkmüş", "Yardım!", "Aman Allah'ım!", "Şimdi ne yapacağım?",
            "Dur!", "Lütfen bırakın beni!", "Korkuyorum!", "Acil yardım!", "Durun!", "Nasıl olur bu?",
            "Yardım edin!", "Eyvah", "korku doluyum", "tedirginim", "ürkütücü", "panik oldum", "ürperdim",
            "titriyorum", "çok korktum", "çok endişeliyim", "ne yapacağımı bilmiyorum", "kaçmam lazım",
            "çok kötü", "bana yardım edin", "burası korkunç", "kaçmak istiyorum", "kalbim duracak", "fearful", "scared",
            "terrified", "anxious", "frightened"
        ],
        "disgust": [
            "tiksinmiş", "iğrenmiş", "tiksindirici", "nefret edici", "iğrenç", "mide bulandırıcı",
            "tiksinti içinde", "rezil", "berbat", "çok kötü", "dayanılmaz", "hoşnutsuz", "rahatsız edici",
            "kabul edilemez", "fena", "bıktım", "bıkkın", "buna dayanamıyorum", "midem bulandı",
            "iğrençsin", "bu ne böyle", "çok tiksindim", "nefret ediyorum", "yeter artık", "tiksiniyorum",
            "mide bulandırıcı", "buna katlanamıyorum", "kötü kokuyor", "kabul edilemez", "çok berbat",
            "çirkin", "pis", "çok kirli", "kirlilik", "mide bulandırıcı", "çok kötü bir durum",
            "nefret ettim", "bunu sevmiyorum", "hiç hoş değil", "nefret ettirici", "mide kaldırmıyor", "disgusted",
            "revolted", "repulsed", "grossed out", "sickened"
        ],
        "neutral": [
            "tarafsız", "nötr", "duygusuz", "duygu belirtisi yok", "sakin", "denge", "normal", "belli değil",
            "kararsız", "orta", "vasat", "ilgisiz", "kayıtsız", "taraf tutmayan", "sıradan", "duygusal değil",
            "tepkisiz", "normal hissediyorum", "rahat", "ne mutlu ne üzgün", "düz", "sakinim", "sakin bir ruh hali",
            "duygu belirtisi yok", "hiçbir şey hissetmiyorum", "hissiz", "rahatım", "karışık değil", "belirsiz",
            "anlamlı değil", "hissiz", "ilgisizim", "rahatım", "gayet sıradan", "duygusuzum", "sade",
            "ne olduğu belli değil",
            "tarafsızım", "düz hissediyorum", "sıradan bir gün", "hiçbir şey yok", "normal bir gün", "rahatım",
            "neutral", "indifferent", "unemotional", "calm", "composed"
        ],
        "excited": [
            "nefes nefese", "istekli", "coşkulu", "heyecanlanmış", "gözleri parlamak", "heyecan verici",
            "kıpır kıpır", "heyecan dolu", "hızlı kalp atışı", "canlılık", "enerjik", "hareketli", "deli gibi mutlu",
            "neşe patlaması", "adrenalin dolu", "heyecan fırtınası", "coşkuyla dolu", "aşırı heyecanlı",
            "tel tel titreyen",
            "ateşli", "hazır ve nazır", "içimde kelebekler", "heyecanla bekliyorum", "süper heyecanlı",
            "çılgınca heyecanlı",
            "hareketlendirici", "doruk noktasında", "adrenalin yüklü", "heyecandan uçuyorum", "heyecan patlaması",
            "yerimde duramıyorum", "enerjik hissediyorum", "hareketliyim", "coşkuyla doluyum", "nefes alamıyorum",
            "yürek hoplatan", "hızlı nefes alıp vermek", "nefes kesen", "zıplamak istiyorum",
            "ayaklarım yerden kesilmiş",
            "canım yerinde durmuyor", "heyecan içindeyim", "ne zaman başlayacak", "zamanın gelmesini bekliyorum",
            "excited", "thrilled", "ecstatic", "enthusiastic", "eager"
        ],
        "confused": [
            "ikilemde", "kafa karışıklığı", "şaşırmış", "kararsız kalmak", "kafası karışmış bir şekilde",
            "yolunu kaybetmiş", "belirsizlik", "ne yapacağımı bilmiyorum", "karışık düşünceler", "anlam veremiyorum",
            "kafası allak bullak", "ne oluyor?", "karışık durum", "kafa duman", "kafa karıştı", "ne yapsam bilemedim",
            "tam bir karmaşa", "ne yapacağını şaşırdı", "bozulmuş", "kafası allak bullak", "zihni karışık",
            "bilmiyorum",
            "ne yapacağını şaşırmış", "ne yapacağını bilemez", "zihni bulanık", "şaşkına dönmüş", "düşünceleri dağınık",
            "ne yapacağını şaşırmış", "kendini kaybetmiş", "ne olacağını bilmiyor", "şaşkına dönmüş", "karar veremiyor",
            "confused", "puzzled", "baffled", "bewildered", "perplexed"
        ],
        "shy": [
            "ürkek bakışlar", "utanmış", "utanarak", "çekingen", "çekinmek", "mahcup", "utangaç bakışlar",
            "endişeli", "gizlenmek", "sıkılgan", "gözlerini kaçırma", "çekinik", "çekingenlik", "saklanmak",
            "mahcubiyet", "çekiniyor", "konuşmak istemiyor", "utangaç bir şekilde", "söze karışmıyor", "sıkılganlık",
            "yüzü kızarmış", "kendini gizlemek", "utanç içinde", "sözünü saklıyor", "içine kapanık", "geri çekilmek",
            "yüzü kızarmış", "sakin", "kendi halinde", "utangaç bir gülümseme", "sessiz", "çekingen tavırlar",
            "utangaç bakışlarla", "sessizce", "sakin sakin", "geri planda", "sessizce konuşmak", "soğuk", "tedirgin",
            "shy", "bashful", "timid", "self-conscious", "reserved"
        ],
        "curious": [
            "sorgulayıcı", "meraklı", "keşfetmeye hevesli", "inceleme yapmak", "bilgi arayan", "keşfetmek isteyen",
            "merak uyandıran", "bilgi düşkünü", "düşündürücü", "keşfetmeye hazır", "fikir sahibi olmak",
            "öğrenmek isteyen", "arayış içinde", "bilgi açlığı", "bilgi sahibi olmak", "bilgi edinmek",
            "araştırma yapmak", "bilgi toplamak", "bilgi yüklü", "araştırmacı", "detaycı", "bilgi sahibi olmak",
            "fikir yürütmek", "bilgili", "keşfetmeye istekli", "bilgi sahibi olmak", "meraklandıran",
            "bilgi paylaşmak", "merak duygusu", "öğrenme isteği", "soru sormak", "bilgiye aç", "bilgi toplamak",
            "curious", "inquisitive", "interested", "inquiring", "exploratory"
        ]
    }

# Metin duygu analizi fonksiyonu
def predict_emotion(metin):
    for duygu, ifadeler in duygu_ifadeleri.items():
        if any(ifade in metin for ifade in ifadeler):
            return duygu
    return "neutral"

@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('deneme2.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    try:
        data = request.get_json()
        selected_emotion = data.get("selected_emotion")
        attempts_left = data.get("attempts_left", 3)

        # Sesin metne dönüştürülmesi
        with sr.Microphone() as source:
            print("Lütfen bir şeyler söyleyin...")
            audio = r.listen(source)
        text = r.recognize_google(audio, language="tr-TR")
        print("Söylediğiniz: " + text)

        # Duygu analizi yap
        emotion = predict_emotion(text)
        print(f"Tahmin edilen duygu: {emotion}, Seçilen duygu: {selected_emotion}")

        if emotion == selected_emotion:
            score = 10
            feedback = "Doğru tahmin! 10 puan kazandınız."
            return jsonify({"transcription": text, "emotion": emotion, "score": score, "feedback": feedback, "attempts_left": attempts_left})
        else:
            attempts_left -= 1
            if attempts_left > 0:
                score = 0
                feedback = "Yanlış tahmin. Tekrar deneyin."
                retry = True
            else:
                score = 0
                feedback = "Yanlış tahmin. Haklarınız doldu."
                retry = False

            return jsonify({"transcription": text, "emotion": emotion, "score": score, "feedback": feedback, "attempts_left": attempts_left, "retry": retry})

    except sr.UnknownValueError:
        return jsonify({"error": "Ne dediğinizi anlayamadım."})

    except sr.RequestError as e:
        return jsonify({"error": "Google Speech Recognition servisine erişilemiyor; {0}".format(e)})

if __name__ == '__main__':
    app.run(debug=True)
