import React, { useEffect } from "react";
import { motion, useAnimation } from "framer-motion";
import { useInView } from "react-intersection-observer";

const skills = [
  {
    name: "Python",
    percent: "90%",
    color: "bg-red-700",
  },
  {
    name: "SQL",
    percent: "95%",
    color: "bg-red-800",
  },
  {
    name: "Cloud Computing",
    percent: "88%",
    color: "bg-red-600",
  },
  {
    name: "Kubernetes",
    percent: "83%",
    color: "bg-rose-600",
  },
  {
    name: "Docker",
    percent: "90%",
    color: "bg-red-800",
  },
  {
    name: "Spark",
    percent: "70%",
    color: "bg-red-400",
  },
  {
    name: "Github",
    percent: "92%",
    color: "bg-red-700",
  },
  {
    name: "CI/CD",
    percent: "94%",
    color: "bg-red-700",
  },
  {
    name: "SnowFlake",
    percent: "85%",
    color: "bg-red-600",
  },
  {
    name: "Frontend",
    percent: "70%",
    color: "bg-amber-600",
  },
];

const Skills = () => {
  const controls = useAnimation();
  const { ref, inView } = useInView({ threshold: 0.1, triggerOnce: true });

  useEffect(() => {
    if (inView) {
      controls.start((i) => ({
        width: skills[i].percent,
        transition: { duration: 1, delay: i * 0.2 },
      }));
    }
  }, [controls, inView]);

  return (
    <>
      <h2 className="font-bold text-5xl w-full text-center p-4">Skills</h2>
      <div className="space-y-4 m-4" ref={ref}>
        {skills.map((skill, index) => (
          <div className="flex items-center justify-between sm:justify-start" key={index}>
            <div className="w-1/3 md:w-1/4 text-right pr-4">
              <span className="text-xs font-bold md:text-lg">{skill.name}</span>
            </div>
            <div className="w-1/2">
              <div className="overflow-hidden h-2 flex rounded bg-gray-300">
                <motion.div
                  initial={{ width: 0 }}
                  animate={controls}
                  custom={index}
                  className={skill.color}
                ></motion.div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </>
  );
};

export default Skills;
