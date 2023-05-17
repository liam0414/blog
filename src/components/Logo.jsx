import React, { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { PlayIcon, PauseIcon } from './Icons';

const Logo = ({ className }) => {
  const [play, setPlay] = useState(false);
  const musicRef = useRef(null);
  const MAX = 20;
  const toggleAudio = () => {
    if (play) {
      musicRef.current?.pause();
      setPlay(false);
    } else {
      musicRef.current?.play();
      setPlay(true);
    }
  };

  const handleVolume = (e) => {
    const { value } = e.target;
    const volume = Number(value) / MAX;
    musicRef.current.volume = volume;
  };
  return (
    <div className={`flex items-center justify-center mt-2 ${className}`}>
      <motion.button
        onClick={toggleAudio}
        className="w-12 h-12 bg-dark text-light flex items-center justify-center rounded-full text-2xl font-bold"
        whileHover={{
          scale: 1.2,
          backgroundColor: [
            '#121212',
            'rgba(131,58,180,1)',
            'rgba(253,29,29,1)',
            'rgba(252,176,69,1)',
            'rgba(131,58,180,1)',
            '#121212'
          ],
          transition: { duration: 1, repeat: Infinity }
        }}>
        {!play ? <PlayIcon /> : <PauseIcon />}
      </motion.button>
      <div className="mx-4 flex">
        <input type="range" className="mr-2 w-full accent-dark" min={0} max={MAX} onChange={(e) => handleVolume(e)} />
      </div>
      <audio ref={musicRef} loop src={'/music.mp3'} />
    </div>
  );
};

export default Logo;
