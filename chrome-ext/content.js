
/**
 * @typedef ProductData
 * @property {string} productName - The name of the product currently being viewed
 * @property {string} productDesc - The description of the product currently being viewed
 */


$(() => {
    /** @type ProductData */
    let product = null;

    if (document.URL.includes('amazon')) {
        product = parseAmazonProduct();
    }
    if (document.URL.includes('walmart')) {
        product = parseWalmartProduct();
    }
    if (document.URL.includes('target')) {
        product = parseTargetProduct();
    }
    
    if (product == null) return;

    console.log(product);
    // ... Do stuff given the product data

});

/**
 * 
 * @returns {ProductData?}
 */
function parseAmazonProduct () {
    // If you are not on an amazon product webpage, return null
    if ( !$('#productTitle').length ) return null;
    // Otherwise, continue parsing the Amazon Product
    const productName = $('#productTitle').text().trim();
    const productDesc = $('#feature-bullets .a-list-item:not(:first)').text().trim();

    return {
        productName,
        productDesc
    }
}

/**
 * @returns {ProductData ?}
 */
function parseWalmartProduct () {
    // If we are not on a walmart product webpage, return null
    if ( !$('h1[itemprop=name]').length ) return null;
    // Otherwise, continue parsing the Walmart Product
    const productName = $('h1[itemprop=name]').text().trim();
    const productDesc = $('.dangerous-html').text().trim();

    return {
        productName,
        productDesc
    }
}

function parseTargetProduct () {
    throw new Exception('Not Implemented Yet');
    // If we are not on a walmart product webpage, return null
    if ( !$('h1[itemprop=name]').length ) return null;
    // Otherwise, continue parsing the Walmart Product
    const productName = $('h1[data-test=product-title]').text().trim();
    const productDesc = "";

    return {
        productName,
        productDesc
    }

}


