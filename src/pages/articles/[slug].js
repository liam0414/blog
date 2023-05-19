import React, { useState, useEffect } from "react";
import { GraphQLClient, gql } from "graphql-request";
import { RichText } from "@graphcms/rich-text-react-renderer";
import { CopyIcon } from "../../components/Icons";
import SyntaxHighlighter from "react-syntax-highlighter";
import { monokai } from "react-syntax-highlighter/dist/cjs/styles/hljs";

const graphcms = new GraphQLClient(process.env.ENDPOINT);
const QUERY = gql`
  query Post($slug: String!) {
    post(where: { slug: $slug }) {
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
      content {
        raw
      }
      categories {
        name
        color {
          css
        }
      }
    }
  }
`;

const SLUGLIST = gql`
  {
    posts {
      slug
    }
  }
`;

export async function getStaticPaths() {
  const { posts } = await graphcms.request(SLUGLIST);
  return {
    paths: posts.map((post) => ({
      params: { slug: post.slug },
    })),
    fallback: false,
  };
}

export async function getStaticProps({ params }) {
  const slug = params.slug;
  const data = await graphcms.request(QUERY, { slug });
  const post = data.post;
  return {
    props: {
      post,
    },
  };
}

const calculateReadingTime = (children) => {
  let textLength = 0;
  const traverse = (child) => {
    if (child.type === "bulleted-list" || child.type === "numbered-list") {
      child.children.forEach((c) => {
        traverse(c);
      });
    } else if (child.type === "list-item") {
      child.children.forEach((c) => {
        traverse(c);
      });
    } else if (child.type === "list-item-child") {
      child.children.forEach((c) => {
        textLength += c.text.trim().split(/\s+/).length;
      });
    } else {
      child.children.forEach((c) => {
        if (c.text !== "") {
          textLength += c.text.trim().split(/\s+/).length;
        }
      });
    }
  };

  children.forEach((child) => {
    traverse(child);
  });

  return textLength;
};

const Article = ({ post }) => {
  const WORDPERMINUTE = 150;
  const readingTime = calculateReadingTime(post.content.raw.children) / WORDPERMINUTE;
  const [copied, setCopied] = useState(false);
  const [showButton, setShowButton] = useState(false);

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleScroll = () => {
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    setShowButton(scrollTop > 0);
  };

  useEffect(() => {
    window.addEventListener("scroll", handleScroll);
    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, []);

  const handleCopy = (e) => {
    navigator.clipboard
      .writeText(e.currentTarget.nextSibling.textContent)
      .then(() => {
        setCopied(true);
        setTimeout(() => {
          setCopied(false);
        }, 2000);
      })
      .catch((error) => {
        console.error("Failed to write to clipboard:", error);
      });
  };
  return (
    <div className="bg-neutral-50">
      <main className="w-3/4 md:w-2/3 lg:w-1/2 flex-col flex items-center mx-auto py-8">
        <div className="flex flex-col w-full">
          <h1 className="text-center w-full text-dark font-bold text-4xl pb-4">{post.title}</h1>
          <div className="flex flex-col gap-2 sm:flex-row justify-between items-center mx-4">
            <p className="flex items-center">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth={1.5}
                stroke="currentColor"
                className="w-6 h-6"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25"
                />
              </svg>
              {`Reading Time: ${Math.ceil(readingTime)} minutes`}
            </p>
            <div className="flex items-center">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth={1.5}
                stroke="currentColor"
                className="w-6 h-6"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M9.568 3H5.25A2.25 2.25 0 003 5.25v4.318c0 .597.237 1.17.659 1.591l9.581 9.581c.699.699 1.78.872 2.607.33a18.095 18.095 0 005.223-5.223c.542-.827.369-1.908-.33-2.607L11.16 3.66A2.25 2.25 0 009.568 3z"
                />
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 6h.008v.008H6V6z" />
              </svg>

              {post.categories.map((tag, index) => (
                <div
                  key={index}
                  style={{ backgroundColor: tag.color.css }}
                  className="inline-flex items-center px-3 py-1 rounded-full text-gray-700 border"
                >
                  {tag.name}
                </div>
              ))}
            </div>
          </div>
        </div>
        {showButton && (
          <button
            className="fixed bottom-5 right-5 p-3 bg-gray-700 text-white rounded-3xl shadow-md"
            onClick={scrollToTop}
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth={1.5}
              stroke="currentColor"
              className="w-6 h-6"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M4.5 12.75l7.5-7.5 7.5 7.5m-15 6l7.5-7.5 7.5 7.5"
              />
            </svg>
          </button>
        )}
        <RichText
          content={post.content.raw.children}
          renderers={{
            p: ({ children }) => (
              <>
                <p className="w-full text-lg">{children}</p>
                <br />
              </>
            ),
            h1: ({ children }) => (
              <h1 className="w-full text-4xl font-bold text-center m-2">{children}</h1>
            ),
            h2: ({ children }) => <h2 className="w-full text-2xl font-bold m-2">{children}</h2>,
            h3: ({ children }) => <h3 className="w-full text-xl font-bold m-2">{children}</h3>,
            img: ({ src, altText }) => (
              <img src={src} alt={altText} className="w-full border border-dark shadow-2xl m-8" />
            ),
            code_block: ({ children }) => {
              return (
                <div className="w-full border border-dark bg-code rounded-xl p-4 m-4">
                  <button
                    className="relative bottom-4 right-4 rounded-lg text-white p-2"
                    onClick={(e) => {
                      handleCopy(e);
                    }}
                  >
                    {copied ? "Copied!" : <CopyIcon />}
                  </button>
                  <SyntaxHighlighter
                    language="javascript"
                    style={monokai}
                    className=" overflow-x-auto  scrollbar-thin scrollbar-thumb-gray-300 scrollbar-track-gray-700 pb-4"
                  >
                    {children.props.content[0].text}
                  </SyntaxHighlighter>
                </div>
              );
            },
            a: ({ href, children }) => (
              <a href={href} className="text-primary hover:text-secondary hover:underline">
                {children}
              </a>
            ),
            blockquote: ({ children }) => <blockquote>{children}</blockquote>,
            ul: ({ children }) => <ul className="w-full pl-8 list-disc m-2 ">{children}</ul>,
            ol: ({ children }) => <ol className="w-full list-decimal pl-8 m-2 ">{children}</ol>,
            li: ({ children }) => <li className="font-serif">{children}</li>,
          }}
        />
      </main>
    </div>
  );
};

export default Article;
