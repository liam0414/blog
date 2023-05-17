import React from "react";
import Head from "next/head";
import AnimatedText from "../components/AnimatedText";
import { GraphQLClient, gql } from "graphql-request";
import "leaflet/dist/leaflet.css";
import dynamic from "next/dynamic";

const Map = dynamic(() => import("../components/Map"), {
  ssr: false,
});

const limit = 10;
const graphcms = new GraphQLClient(process.env.ENDPOINT);

export async function getStaticProps() {
  const QUERY = gql`
    query {
      travels(first: 100) {
        destination
        photo {
          url
        }
        slug
        latlng {
          latitude
          longitude
        }
      }
    }
  `;

  const { travels } = await graphcms.request(QUERY);
  return {
    props: {
      travels,
    },
  };
}

const Travels = ({ travels }) => {
  return (
    <>
      <Head>
        <title>Travels</title>
        <meta name="my trips" content="my trips" />
      </Head>
      <main className="w-full flex flex-col items-center justify-center m-auto">
        <AnimatedText text="A thousand miles begins with a single step" />
        <Map travels={travels} />
      </main>
    </>
  );
};

export default Travels;
