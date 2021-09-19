
let urlBase = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json?';
let geolocationPosition;
let topRetailers = ["walmart","amazon","the kroger co.","home depot","costco","walgreens","target","cvs","lowe's","albertsons companies","apple","royal ahold delhaize usa","publix super markets","best buy","aldi","dollar general","h.e. butt grocery","tjx companies","dollar tree","ace hardware","meijer","wakefern","shoprite","7-eleven","macy's","at&amp;t wireless","rite aid","verizon wireless","bj's wholesale club","kohl's","petsmart","menards","ross stores","dell","hy vee","wayfair","o'reilly auto parts","gap","qurate retail","health mart systems","wegmans food market","l brands","autozone","tractor supply co.","giant eagle","alimentation couche-tard","dick's sporting goods","sherwin-williams","nordstrom","bed bath &amp; beyond","winco foods","good neighbor pharmacy","southeastern grocers (bi-lo)","army &amp; air force exchange","j.c. penney","save-a-lot","bass pro","staples","williams-sonoma","sprouts farmers market","speedway","avb brandsource","big lots","ulta salon, cosmetics &amp; fragrance","foot locker","ikea north american services","office depot","academy sports","burlington","camping world","discount tire","sephora","true value co.","piggly wiggly","hobby lobby stores","petco","michaels stores","stater bros holdings","signet jewelers","exxon mobile corporation","defense commissary agency","my demoulas","advance auto","dillard's","smart &amp; final","weis markets","ingles","golub","shell oil company","save mart","total wine &amp; more","caseys general store","guitar center","gamestop","american eagle","unfi","grocery outlet","belk","lululemon","sears","ampm"];

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
        let closePlaces = [...new Set(data.results
                                        .filter(place => !topRetailers.includes(place.name.toLowerCase()))
                                        .map(place => place.name + ": " + place.vicinity)
                                        )];
                                        
        let divPlaces = document.getElementById("divPlaces");

        divPlaces.classList.toggle('hidden');

        for (let place of closePlaces.slice(0,2)) {
            divPlaces.innerHTML += place + "\n";
        }
    })
}