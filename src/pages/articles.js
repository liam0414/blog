import React, { useState } from "react";
import Head from "next/head";
import AnimatedText from "../components/AnimatedText";
import { GraphQLClient, gql } from "graphql-request";
import Article from "../components/Article";

const graphcms = new GraphQLClient(process.env.ENDPOINT);
const QUERY = gql`
  {
    posts(first: 100) {
      title
      author {
        name
        avatar {
          url
        }
      }
      coverPhoto {
        url
      }
      createdAt
      slug
      description
      categories {
        name
        color {
          css
        }
      }
    }
  }
`;

export async function getStaticProps() {
  const { posts } = await graphcms.request(QUERY);
  return {
    props: {
      posts,
    },
  };
}

const Articles = ({ posts }) => {
  const [filterTags, setFilterTags] = useState([]);
  const filterPosts = (e) => {
    const filterString = e.currentTarget.textContent;
    if (!filterTags.includes(filterString)) {
      setFilterTags((prev) => prev.concat(filterString));
    }
  };
  const removeFilter = (e) => {
    const filterString = e.currentTarget.textContent;
    setFilterTags((prev) => prev.filter((tag) => tag !== filterString));
  };
  return (
    <>
      <Head>
        <title>Articles</title>
        <meta name="my articles" content="my articles" />
      </Head>

      <main
        className="w-full flex flex-col items-center justify-center"
        style={{
          backgroundImage: 'url("/images/background2.png")',
          backgroundRepeat: "no-repeat",
          backgroundSize: "cover",
          backgroundPosition: "center",
        }}
      >
        <AnimatedText text="Live and learn" />
        <div>
          {filterTags.map((tag, index) => (
            <div
              key={index}
              onClick={removeFilter}
              className="hover:cursor-pointer ml-4 text-xs inline-flex items-center font-bold leading-sm uppercase px-3 py-1 rounded-full bg-white text-gray-700 border"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth={1.5}
                stroke="currentColor"
                className="w-6 h-6"
              >
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
              {tag}
            </div>
          ))}
        </div>

        <div className="w-full grid gap-4 grid-cols-1 md:grid-cols-2 lg:grid-cols-3 p-8">
          {filterTags.length === 0 &&
            posts.map((post, index) => (
              <Article post={post} filterPosts={filterPosts} key={index} />
            ))}
          {filterTags.length !== 0 &&
            posts
              .filter((post) => {
                const matchedTags = post.categories.map((category) => category.name);
                return filterTags.every((tag) => matchedTags.includes(tag));
              })
              .map((post, index) => <Article post={post} filterPosts={filterPosts} key={index} />)}
        </div>
      </main>
    </>
  );
};

export default Articles;
