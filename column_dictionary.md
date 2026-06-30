# CRMLS Sold Dataset — Key Columns

Data source: CRMLS (California Regional Multiple Listing Service) sold listings (RESO Data Dictionary field names). These are the columns most useful for analysis/modeling.

## Price & sale outcome
| Column | Meaning |
|--------|---------|
| ClosePrice | Final sale price — the main target/outcome variable |
| ListPrice | Most recent list price |
| OriginalListPrice | Price the property was first listed at |
| DaysOnMarket | Days the listing was active before going under contract |

## Property characteristics
| Column | Meaning |
|--------|---------|
| PropertyType | Top-level type (Residential, Commercial, Land…) |
| PropertySubType | Detailed type (Single Family Residence, Condominium…) |
| LivingArea | Finished living area (sq ft) |
| LotSizeSquareFeet | Lot size in square feet |
| BedroomsTotal | Total bedrooms |
| BathroomsTotalInteger | Total bathrooms (whole number) |
| YearBuilt | Year built |
| Stories | Number of stories |
| GarageSpaces | Garage parking spaces |
| PoolPrivateYN | Has a private pool (Yes/No) |
| ViewYN | Has a view (Yes/No) |
| FireplacesTotal | Number of fireplaces |
| NewConstructionYN | New construction (Yes/No) |

## Location
| Column | Meaning |
|--------|---------|
| City | City |
| PostalCode | ZIP code |
| CountyOrParish | County |
| Latitude | Property latitude |
| Longitude | Property longitude |
| UnparsedAddress | Full street address |
| MLSAreaMajor | Major MLS geographic area |

## Costs / taxes
| Column | Meaning |
|--------|---------|
| TaxAnnualAmount | Annual property tax |
| AssociationFee | HOA fee amount |
| AssociationFeeFrequency | HOA fee billing frequency (Monthly, Annually…) |

## Timing
| Column | Meaning |
|--------|---------|
| CloseDate | Date the sale closed |
| ListingContractDate | Date listed for sale |
| PurchaseContractDate | Date the offer was accepted (went into escrow) |

## Identifiers
| Column | Meaning |
|--------|---------|
| ListingId | Human-readable MLS listing number |
| ListingKey | Unique system ID for the listing record |
