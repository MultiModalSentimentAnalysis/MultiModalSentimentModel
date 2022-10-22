import cv2
import pysrt
import numpy as np
from math import floor, ceil
from time import time
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip

from moviepy.video.tools.subtitles import SubtitlesClip

for id in [1]:
# for movie_name, sub_name in [("movie_1.mkv", "sub_1.srt")]:
    movie_name = f"movie_{id}.mkv"
    sub_name = f"sub_{id}.srt"
    s = time()
    subs = pysrt.open(sub_name)
    movie = VideoFileClip(movie_name)
    duration = movie.duration
    number_of_samples = floor(duration / 60) # number of samples ~ length of video in minutes
    step = floor(len(subs) / number_of_samples)
    sub_indexes = np.arange(number_of_samples) * step + 1

    for sub_index in sub_indexes:
        sub = subs[sub_index-1]
        text = sub.text
        if not text.isascii():
            continue
        start = sub.start
        end = sub.end
        start = start.hours * 3600 + start.minutes * 60 + start.seconds + start.milliseconds/1000
        end = end.hours * 3600 + end.minutes * 60 + end.seconds + end.milliseconds/1000
        # start = floor(start.hours * 3600 + start.minutes * 60 + start.seconds + start.milliseconds/1000)
        # end = ceil(end.hours * 3600 + end.minutes * 60 + end.seconds + end.milliseconds/1000)

        # clip = VideoFileClip(movie_name).subclip(start, end)
        # clip.to_videofile("1.mkv", temp_audiofile='temp-audio.m4a', remove_temp=True, audio_codec='aac')
        # clip.to_videofile("1.mkv", codec="libx264", temp_audiofile='temp-audio.m4a', remove_temp=True, audio_codec='aac')
        # ffmpeg_extract_subclip(movie_name, 1030.9, 1031, targetname=f"1.mkv")
        name = "movie_{id}_clip_{sub_index}.mkv"
        ffmpeg_extract_subclip(movie_name, start, end, targetname=name)
        break

    # from ffpyplayer.player import MediaPlayer
    # video=cv2.VideoCapture(name)
    # player = MediaPlayer(name)
    # while True:
    #     print('in while')
    #     grabbed, frame=video.read()
    #     audio_frame, val = player.get_frame()
    #     if not grabbed:
    #         print("End of video")
    #         break
    #     if cv2.waitKey(25) & 0xFF == ord("q"):
    #         break
    #     cv2.imshow("Video", frame)
    #     if val != 'eof' and audio_frame is not None:
    #         #audio
    #         img, t = audio_frame
    # video.release()
    # cv2.destroyAllWindows()

    cap = cv2.VideoCapture(name)
    player = MediaPlayer(name)
    if (cap.isOpened()== False):
        print("Error opening video file")
    
    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        audio_frame, val = player.get_frame()
        if ret == True:
        # Display the resulting frame
            cv2.imshow('Frame', frame)
            # if val != 'eof' and audio_frame is not None:
            #     #audio
            #     img, t = audio_frame
            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    
    # Break the loop
        else:
            break
    
    # # When everything done, release
    # # the video capture object
    # cap.release()
    
    # # Closes all the frames
    # cv2.destroyAllWindows()


    print(time()-s)













# For sub in tagging
    # generator = lambda txt: TextClip(txt, font='Arial', fontsize=24, color='white')
    # def sub_to_list(sub):
    #     text = sub.text
    #     start = sub.start.minutes * 60 + sub.start.seconds
    #     end = sub.end.minutes * 60 + sub.end.seconds
    #     return (start, end), text

    # subs_list = list(map(sub_to_list, subs))
    # subtitles = SubtitlesClip(subs_list, generator)
    # result = CompositeVideoClip([movie, subtitles.set_pos(('center','bottom'))])

    # result.write_videofile("output.mp4", fps=subtitles.fps, temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
    # exit()