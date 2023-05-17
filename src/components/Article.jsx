import React from "react";
import Link from "next/link";

const Article = ({ post, filterPosts }) => {
  const { title, author, coverPhoto, createdAt, slug, description, categories } = { ...post };

  return (
    <article className="w-full flex flex-col items-center rounded-3xl border border-solid border-dark bg-card p-4 relative">
      <Link href={"/articles/" + slug} className="w-full flex justify-center">
        <img src={coverPhoto.url} alt={title} />
      </Link>
      <Link href={"/articles/" + slug} className="hover:underline underline-offset-2 h-28">
        <h2 className="m-2 text-left text-lg md:text-2xl lg:text-3xl font-bold hover:text-primary">
          {title}
        </h2>
      </Link>
      <div className="flex justify-center space-x-4 mb-4">
        {categories.map((category, index) => (
          <div
            className="hover:cursor-pointer p-2 rounded-xl uppercase"
            style={{ backgroundColor: category.color.css }}
            key={index}
            onClick={filterPosts}
          >
            {category.name}
          </div>
        ))}
      </div>
      <div className="w-3/4 hidden sm:block">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <img src={author.avatar.url} alt="author" className="w-12 h-12 rounded-full" />
            <div className="text-white">{author.name}</div>
          </div>
          <div className="text-white">{createdAt.substring(0, 10)}</div>
        </div>
      </div>
      <p className="text-white font-semibold hidden sm:block p-4">{description}</p>
    </article>
  );
};

export default Article;
