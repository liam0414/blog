import React, { useRef } from 'react';
import { motion, useScroll } from 'framer-motion';
import LiIcon from '@/components/LiIcon';

const JobDetails = ({ position, company, time, address, work }) => {
  const ref = useRef(null);
  return (
    <li ref={ref} className="my-8 first:mt-0 last:mb-0 w-[60%] mx-auto flex flex-col items-start jusitfy-between">
      <LiIcon reference={ref} />
      <motion.div initial={{ y: 50 }} whileInView={{ y: 0 }} transition={{ duration: 0.5, type: 'spring' }}>
        <h3 className="capitalize font-bold text-xl md:text-2xl">
          {position}&nbsp;<div className="text-primary">@{company}</div>
        </h3>
        <span className="capitalize font-medium text-dark/75">
          {time} | {address}
        </span>
      </motion.div>
    </li>
  );
};

const Experience = () => {
  const ref = useRef(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ['start end', 'center start']
  });
  return (
    <div className="my-16">
      <h2 className="font-bold text-4xl md:text-6xl mb-16 w-full text-center">Experience</h2>
      <div ref={ref} className="w-[75%] mx-auto relative">
        <motion.div
          style={{ scaleY: scrollYProgress }}
          className="absolute left-9 top-2 w-[4px] h-full bg-dark origin-top"
        />
        <ul className="w-full flex flex-col items-start justify-start ml-4">
          <JobDetails
            position="Data Engineer Graduate"
            company="Quantium"
            time="2023-present"
            address="Sydney, Australia"
          />
          <JobDetails
            position="Platform Engineer Intern"
            company="Quantium"
            time="2022-2023"
            address="Sydney, Australia"
          />
          <JobDetails position="Techinical Engineer" company="TfNSW" time="2020-2022" address="Sydney, Australia" />
        </ul>
      </div>
    </div>
  );
};

export default Experience;
