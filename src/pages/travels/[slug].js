import React, { useState } from 'react';
import { GraphQLClient, gql } from 'graphql-request';
import { motion } from 'framer-motion';
import Image from 'next/image';
const graphcms = new GraphQLClient(process.env.ENDPOINT);

const QUERY = gql`
  query Travel($slug: String!) {
    travel(where: { slug: $slug }) {
      destination
      photo {
        url
      }
      slug
      year
    }
  }
`;

const SLUGLIST = gql`
  {
    travels {
      slug
    }
  }
`;

export async function getStaticPaths() {
  const { travels } = await graphcms.request(SLUGLIST);
  return {
    paths: travels.map((travel) => ({
      params: { slug: travel.slug }
    })),
    fallback: false
  };
}

export async function getStaticProps({ params }) {
  const slug = params.slug;
  const data = await graphcms.request(QUERY, { slug });
  const travel = data.travel;
  return {
    props: {
      travel
    }
  };
}

const Modal = ({ images, onClose, index }) => {
  const [currentIndex, setCurrentIndex] = useState(index);

  const handlePrev = (event) => {
    event.stopPropagation();
    setCurrentIndex((prevIndex) => (prevIndex === 0 ? 0 : prevIndex - 1));
  };

  const handleNext = (event) => {
    event.stopPropagation();
    setCurrentIndex((prevIndex) => (prevIndex === images.length - 1 ? images.length - 1 : prevIndex + 1));
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.8 }}
      className="fixed flex bg-black bg-opacity-75 justify-center items-center w-full p-4 md:inset-0 h-[calc(100%-1rem)] max-h-full"
      onClick={onClose}>
      <motion.div
        initial={{ opacity: 0, y: 0 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: 0 }}
        className="relative w-full max-w-lg max-h-full">
        <div className="relative">
          <button
            className="absolute top-1/2 -left-8 transform -translate-y-1/2 bg-white rounded-full p-2 shadow-md opacity-75 hover:opacity-100 focus:opacity-100 transition-opacity"
            onClick={handlePrev}>
            &lt;
          </button>
          <button
            className="absolute top-1/2 -right-8 transform -translate-y-1/2 bg-white rounded-full p-2 shadow-md opacity-75 hover:opacity-100 focus:opacity-100 transition-opacity"
            onClick={handleNext}>
            &gt;
          </button>
          <Image src={images[currentIndex].url} alt="focusedImg" className="rounded-xl" />
        </div>
      </motion.div>
    </motion.div>
  );
};

const Trip = ({ travel }) => {
  const [modal, setModal] = useState(false);
  const [modalImg, setModelImg] = useState(0);

  const openModal = (url, index) => {
    setModelImg(index);
    setModal(true);
  };

  const closeModal = () => {
    setModal(false);
  };

  return (
    <>
      {modal && <Modal images={travel.photo} onClose={closeModal} index={modalImg} />}
      <div className="px-32">
        <h1 className="text-center text-4xl font-bold text-gray-500 py-8">
          {travel.destination.charAt(0).toUpperCase() + travel.destination.slice(1)} - {travel.year}
        </h1>
        <div className="sm:columns-2 md:columns-3 lg:columns-4 gap-3 mx-auto space-y-3 pb-28">
          {travel.photo.map((p, index) => {
            return (
              <div className="bg-gray-200 break-inside-avoid " key={index}>
                <Image
                  src={p.url}
                  alt={travel.destination + index}
                  className="rounded-lg hover:cursor-pointer"
                  onClick={() => openModal(p.url, index)}
                />
              </div>
            );
          })}
        </div>
      </div>
    </>
  );
};

export default Trip;
