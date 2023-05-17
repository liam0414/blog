import React from 'react';
import Link from 'next/link';
import { Marker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

const icon = new L.icon({
  iconUrl: '/images/marker-icon.png',
  shadowUrl: './images/marker-shadow.png',
  iconAnchor: [6, 40],
  popupAnchor: [7, -30]
});

const MyMarker = ({ trip }) => {
  const map = useMap();
  return (
    <Marker position={[trip.latlng.latitude, trip.latlng.longitude]} icon={icon}>
      <Popup>
        <Link href={'/travels/' + trip.slug}>
          {trip.destination.charAt(0).toUpperCase() + trip.destination.slice(1)}
        </Link>{' '}
      </Popup>
    </Marker>
  );
};

export default MyMarker;
