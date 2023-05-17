import React, { useState } from "react";
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

const Article = ({ post }) => {
  const [copied, setCopied] = useState(false);
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
    <div className="bg-neutral-200">
      <main className="w-3/4 md:w-2/3 lg:w-1/2 flex-col flex items-center mx-auto py-8">
        <RichText
          content={post.content.raw.children}
          renderers={{
            p: ({ children }) => <p className="w-full text-lg">{children}</p>,
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
            ul: ({ children }) => <ul className="w-full pl-8 list-disc m-2">{children}</ul>,
            ol: ({ children }) => <ol className="w-full list-decimal pl-8 m-2">{children}</ol>,
            li: ({ children }) => <li>{children}</li>,
          }}
        />
      </main>
    </div>
  );
};

export default Article;
