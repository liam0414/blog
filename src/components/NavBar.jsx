import React, { useState } from "react";
import Link from "next/link";
import Logo from "./Logo";

const CustomLink = ({ href, title, className = "" }) => {
  return (
    <Link href={href} className={`${className} relative group text-dark`}>
      {title}
      <span
        className="h-[3px] inline-block w-0 bg-dark absolute left-0 -bottom-0.5 
          group-hover:w-full transition-[width] ease duration-300"
      >
        &nbsp;
      </span>
    </Link>
  );
};
const links = ["Projects", "Travels", "Articles"];
const NavBar = () => {
  const [isOpen, setIsOpen] = useState(false);

  const handleClick = () => {
    setIsOpen((prev) => !prev);
  };
  return (
    <header className="w-full px-4 md:px-16 py-8 font-medium ">
      <div className="hidden md:flex items-center justify-between">
        <Logo />
        <nav>
          <CustomLink href="/" title="Home" className="mr-2" />
          <CustomLink href="/projects" title="Projects" className="mx-2" />
          <CustomLink href="/travels" title="Travels" className="mx-2" />
          <CustomLink href="/articles" title="Articles" className="mx-2" />
        </nav>
      </div>

      <div>
        <button
          className="flex flex-col justify-center items-center md:hidden"
          onClick={handleClick}
        >
          <span
            className={`bg-dark block h-1 w-8 rounded-sm ${
              isOpen ? "rotate-45 translate-y-0.5" : "-translate-y-1"
            }`}
          ></span>
          <span className={`bg-dark block h-1 w-8 rounded-sm ${isOpen ? "hidden" : ""}`}></span>
          <span
            className={`bg-dark block h-1 w-8 rounded-sm ${
              isOpen ? "-rotate-45 -translate-y-0.5" : "translate-y-1"
            }`}
          ></span>
        </button>
        {isOpen && (
          <div className="absolute mt-4 z-10 border bg-white divide-y divide-gray-100 w-32 dark:bg-gray-700 dark:divide-gray-600">
            <Link
              href="/"
              className="text-gray-700 hover:bg-gray-700 hover:text-white block px-2 py-2 text-base font-medium"
            >
              Home
            </Link>
            {links.map((link, index) => (
              <Link
                key={index}
                href={link.toLowerCase()}
                className="text-gray-700 hover:bg-gray-700 hover:text-white block px-2 py-2 text-base font-medium"
              >
                {link}
              </Link>
            ))}
          </div>
        )}
      </div>
    </header>
  );
};

export default NavBar;
