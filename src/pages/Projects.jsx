import React, { useState, useEffect } from 'react';
import { Octokit } from 'octokit';
import Image from 'next/image';
const Projects = () => {
  const octokit = new Octokit({
    auth: process.env.TOKEN
  });

  const [repos, setRepos] = useState([]);

  useEffect(() => {
    octokit
      .request('GET /users/{username}/repos', {
        username: 'ligam0414',
        headers: {
          'X-GitHub-Api-Version': '2022-11-28'
        }
      })
      .then((res) => setRepos(res.data));
  }, []);

  return (
    <div className="container mx-auto px-4 py-8 sm:px-6 lg:px-8">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 md:grid-cols-2 lg:grid-cols-3">
        {repos.map((repo, index) => (
          <div
            key={index}
            className="w-full relative bg-white border border-gray-200 rounded-lg shadow dark:bg-gray-800 dark:border-gray-700">
            <a href={repo.html_url}>
              <Image className="rounded-t-lg" src="/images/projects/devdreaming.jpg" alt="" />
            </a>
            <div className="p-5">
              <a href={repo.html_url}>
                <h5 className="mb-2 text-2xl font-bold tracking-tight text-gray-900 dark:text-white">{repo.name}</h5>
              </a>
              {repo.description && (
                <div className="pb-8">
                  <p className="truncate overflow-hidden mb-3 font-normal text-gray-700 dark:text-gray-400 hover:whitespace-normal">
                    {repo.description}
                  </p>
                </div>
              )}
              {!repo.description && <p className="mb-8 font-normal text-gray-700 dark:text-gray-400">No Description</p>}
              <a
                href={repo.html_url}
                className="absolute bottom-2 inline-flex items-center px-3 py-2 text-sm font-medium text-center text-white bg-blue-700 rounded-lg hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300 dark:bg-blue-600 dark:hover:bg-blue-700 dark:focus:ring-blue-800">
                Read more
                <svg
                  aria-hidden="true"
                  className="w-4 h-4 ml-2 -mr-1"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                  xmlns="http://www.w3.org/2000/svg">
                  <path
                    fillRule="evenodd"
                    d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z"
                    clipRule="evenodd"></path>
                </svg>
              </a>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Projects;
