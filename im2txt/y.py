#-*- coding: UTF-8 -*-
import sys
import pyttsx3
import importlib
importlib.reload(sys)


engine = pyttsx3.init()
#语速控制
rate = engine.getProperty('rate')
engine.setProperty('rate', rate-60)
#音量控制
volume = engine.getProperty('volume')
engine.setProperty('volume', volume-0.9)
#更换发音人声音
voices = engine.getProperty('voices')
for voice in voices:
   engine.setProperty('voice', voice.id)
   engine.say('thunder')
   engine.say('二零一八枪毙名单点名开始')
engine.runAndWait()
#self._tts.Speak(fromUtf8(toUtf8(text)), 19)
