
let urlBase = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json?';
let geolocationPosition;

window.addEventListener('load', (event) => {
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition((position) => {
            geolocationPosition = position;
            document.getElementById("buttonSearch").classList.toggle('hidden');
            document.getElementById("loadingDiv").classList.toggle('hidden');
        })
    } else {
        console.log("Geolocation services unavailable");
    }
});

document.getElementById("buttonSearch").addEventListener('click', searchNearby);

function searchNearby() {
    let url = urlBase 
                + 'location=' + geolocationPosition.coords.latitude + '%2c' + geolocationPosition.coords.longitude
                + '&radius=3000'
                + '&type=convenience_store'
                + '&key=AIzaSyAnsQx-Ckzs079P05WIdwik8ocdMgTOCoI';

    fetch(url)
    .then(response => response.json())
    .then(data => {
        let closePlaces = [...new Set(data.results.map(place => place.name + ": " + place.vicinity))];
        console.log(closePlaces);


        let divPlaces = document.getElementById("divPlaces");

        divPlaces.classList.toggle('hidden');

        for (let place of closePlaces.slice(0,2)) {
            divPlaces.innerHTML += place + "\n";
        }
    })
}