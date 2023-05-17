import React, { useEffect, useRef } from 'react';
import { useInView, useMotionValue, useSpring } from 'framer-motion';
import Link from 'next/link';
const AnimatedNumbers = ({ value }) => {
  const ref = useRef(null);
  const motionValue = useMotionValue(0);
  const springValue = useSpring(motionValue, { duration: 2000 });
  const isInView = useInView(ref);

  useEffect(() => {
    if (isInView) {
      motionValue.set(value);
    }
  }, [isInView, motionValue, value]);

  useEffect(() => {
    springValue.on('change', (latest) => {
      if (ref.current && latest.toFixed(0) <= value) {
        ref.current.textContent = latest.toFixed(0);
      }
    });
  }, [springValue, value]);

  return <span ref={ref}></span>;
};

const Statistics = () => {
  const countries = ['China', 'Australia', 'Japan', 'New Zealand'];
  const years = new Date().getFullYear() - 2020;
  return (
    <div className="p-16 mb-4 col-span-2 md:flex md:flex-row items-center md:justify-between">
      <Link href="/projects">
        <div className="flex flex-col items-center justify-center p-2">
          <span className="inline-block text-6xl md:text-5xl font-bold hover:cursor-pointer">
            <AnimatedNumbers value={10} />
          </span>
          <h2 className="font-medium capitalize text-dark/75">Projects Completed</h2>
        </div>
      </Link>
      <Link href="/travels">
        <div className="flex flex-col items-center justify-center p-2">
          <span className="inline-block text-6xl md:text-5xl font-bold hover:cursor-pointer">
            <AnimatedNumbers value={countries.length} />
          </span>
          <h2 className="font-medium capitalize text-dark/75">Countries Travelled</h2>
        </div>
      </Link>
      <Link href="/experience">
        <div className="flex flex-col items-center justify-center p-2">
          <span className="inline-block text-6xl md:text-5xl font-bold hover:cursor-pointer">
            <AnimatedNumbers value={years} />
          </span>
          <h2 className="font-medium capitalize text-dark/75">Years of Experience</h2>
        </div>
      </Link>
    </div>
  );
};

export default Statistics;
