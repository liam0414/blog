import React from 'react';
import { MapContainer, TileLayer } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import MyMarker from './MyMarker';
import MarkerClusterGroup from '@changey/react-leaflet-markercluster';

const Map = ({ travels }) => {
  const defaultPos = [0, 151.2093];
  return (
    <MapContainer
      className="w-full sm:w-3/4 md:w-2/3 lg:w-1/2 h-96 m-4"
      center={defaultPos}
      zoom={1}
      scrollWheelZoom={false}>
      <TileLayer
        attribution='&copy; <a href="https://carto.com/">carto.com</a> contributors'
        url="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png"
      />
      <MarkerClusterGroup showCoverageOnHover={false}>
        {travels.map((trip, index) => (
          <MyMarker trip={trip} key={index} />
        ))}
      </MarkerClusterGroup>
    </MapContainer>
  );
};

export default Map;
