import React from "react";
import Layout from "./Layout";
import { GithubIcon, YoutubeIcon, LinkedInIcon } from "./Icons";
import { motion } from "framer-motion";

const Footer = () => {
  return (
    <footer className="w-full font-medium">
      <Layout className="py-16 flex flex-col items-center text-center space-y-6 bg-dark text-light">
        <h1 className="text-3xl">SUBSCRIBE</h1>
        <p>Sign up with your email address to receive new content.</p>
        <div className="max-w-7xl mx-auto py-4 sm:px-6 lg:px-8 text-dark">
          <div className="grid grid-cols-1 md:grid-cols-7 gap-2 ">
            <div className="col-span-2">
              <input
                id="first"
                type="text"
                placeholder="first name"
                className="h-10 block w-full rounded-md pl-2"
              />
            </div>
            <div className="col-span-2">
              <input
                id="last"
                type="text"
                placeholder="last name"
                className="h-10 block w-full rounded-md pl-2"
              />
            </div>
            <div className="col-span-2">
              <input
                id="email"
                type="email"
                autoComplete="off"
                placeholder="email"
                className="h-10 block w-full rounded-md pl-2"
              />
            </div>
            <div>
              <button
                type="submit"
                className=" py-2 px-6 bg-green-500 text-dark font-medium rounded-md shadow-md"
              >
                Submit
              </button>
            </div>
          </div>
        </div>

        <nav className="flex items-center justify-center flex-wrap space-x-2">
          <motion.a
            aria-label="github"
            href="https://github.com/ligam0414"
            target={"_blank"}
            whileHover={{ y: -4 }}
            whileTap={{ scale: 0.9 }}
            className="w-6"
          >
            <GithubIcon />
          </motion.a>
          <motion.a
            aria-label="youtube"
            href="https://www.youtube.com/channel/UC32ldb54TTW_ObW1X167Ubg"
            target={"_blank"}
            whileHover={{ y: -4 }}
            whileTap={{ scale: 0.9 }}
            className="w-6"
          >
            <YoutubeIcon />
          </motion.a>
          <motion.a
            aria-label="linkedin"
            href="https://www.linkedin.com/in/liam-c-625565170/"
            target={"_blank"}
            whileHover={{ y: -4 }}
            whileTap={{ scale: 0.9 }}
            className="w-6"
          >
            <LinkedInIcon />
          </motion.a>
        </nav>
        <span>{new Date().getFullYear()} &copy; All Rights Reserved</span>
      </Layout>
    </footer>
  );
};

export default Footer;
