# Install and import the libraries
library(ggplot2)
library(dplyr)
#install.packages("plm")
#install.packages("lmtest")
#install.packages("sandwich")
#install.packages("car")
library(plm)
library(lmtest)
library(sandwich)
library(car)
#install.packages("tidyr")
library(tidyr)

panel_df <- read.csv('../output_data/panel_dataset.csv')

# Define the policy variables
policy_vars <- c("Mitigation", "Adaptation", "Disaster_Risk_Management", "Loss_and_Damage")

# Define dependent variables: all columns after 'Year' excluding policy vars
dep_vars <- setdiff(colnames(panel_df), c("Country", "Country_Name", "Year", policy_vars))

# Select only numeric dependent variables
numeric_dep_vars <- dep_vars[sapply(panel_df[, dep_vars], is.numeric)]

# Select only numeric policy variables
numeric_policy_vars <- policy_vars[sapply(panel_df[, policy_vars], is.numeric)]

# Standardize only numeric vars
panel_df_std <- panel_df %>%
  mutate(across(all_of(c(numeric_dep_vars, numeric_policy_vars)), ~ scale(.)))

## Function to interpret VIF
interpret_vif <- function(vif_values) {
  for (var in names(vif_values)) {
    v <- vif_values[var]
    message <- if (v == 1) {
      "No multicollinearity exists."
    } else if (v > 1 && v < 5) {
      "Moderate multicollinearity. But it is acceptable."
    } else if (v >= 5 && v <= 10) {
      "High multicollinearity. No acceptable."
    } else {
      "Severe multicollinearity. No acceptable."
    }
    cat(sprintf("Variable: %s, VIF: %.2f — %s\n", var, v, message))
  }
}

results_list <- list()

## Run this function to check all the variables that test significant against our policy variable
for (dv in dep_vars) {
  cat("\nChecking variable:", dv)
  
  # Skip non-numeric DVs
  if (!is.numeric(panel_df_std[[dv]])) {
    cat("  → Skipped: not numeric.\n")
    next
  }
  
  # Create a subset for this DV + policy vars
  vars_for_model <- c("Country", "Year", policy_vars, dv)
  sub_df <- panel_df_std[, vars_for_model]
  
  # Remove rows with NA in dependent OR policy variables only
  sub_df <- sub_df[complete.cases(sub_df[, c(dv, policy_vars)]), ]
  
  # Skip if too few rows remain
  if (nrow(sub_df) < 10) {
    print("Skipped: too few rows after NA removal.\n")
    next
  }
  
  # Check for within-country variation in DV
  within_var <- sub_df %>%
    group_by(Country) %>%
    summarise(var = var(.data[[dv]], na.rm = TRUE)) %>%
    pull(var)
  
  if (all(within_var == 0 | is.na(within_var))) {
    print("Skipped: no within-country variation.\n")
    next
  }
  
  # Fit the fixed effects model
  formula_obj <- as.formula(paste0(dv, " ~ ", paste(policy_vars, collapse = " + ")))
  
  ## Fixed effects
  fit <- tryCatch({
    plm(formula_obj, data = sub_df, index = c("Country", "Year"), model = "within", effect = "twoways")
  }, error = function(e) {
    cat("FE model error:", conditionMessage(e), "\n")
    return(NULL)
  })

  
  # Check coefficients and filter for significance
  if (!is.null(fit)) {
    coefs <- summary(fit)$coefficients
    
    if (!is.null(coefs)) {
      sig <- coefs[
        rownames(coefs) %in% policy_vars & coefs[, "Pr(>|t|)"] < 0.05,
        , drop = FALSE
      ]
      
      if (nrow(sig) > 0) {
        print("Significant result found.")
        ## For addition to the dataframe
        sig_result <- data.frame(
          Dependent_Var = dv,
          Policy_Var = rownames(sig),
          Estimate = sig[, "Estimate"],
          Std_Error = sig[, "Std. Error"],
          P_Value = sig[, "Pr(>|t|)"],
          row.names = NULL
        )
        results_list[[length(results_list) + 1]] <- sig_result
        
        cat('\n Performing diagnostic tests ')
        
        ## Check multicollinearity using Variance Inflation Factor to detect multicollinearity among policy variables
        cat("\n Checking for multicollinearity.")
        vif_vals <- vif(lm(formula_obj, data = sub_df))
        print(vif_vals)
        interpret_vif(vif_vals)
        
        ## Homoskedasticity
        bp_test <- bptest(fit)
        cat('\n Breusch-Pagan Test')
        print(bp_test)
        
        ## Autocorrelation
        serial_test <- pbgtest(fit)
        cat('\nSerial Correlation (Panel BG) Test')
        print(serial_test)
        
        
      } else {
        print("No significant policy variables.")
        ## just keep it empty if no need to print for all the variables where significance was not observed.
      }
    }
  }
}

# Combine and display results
# If results are found, combine them to a dataframe
results_df <- if (length(results_list) > 0) do.call(rbind, results_list) else data.frame() 
## to include the confidence intervals
results_df <- results_df %>%
  mutate(
    CI_Lower = Estimate - 1.96 * Std_Error,
    CI_Upper = Estimate + 1.96 * Std_Error
  )
print("\nFinal significant results:\n")
print(results_df)


### Given the results we got, first we map them to their series name and create a shorter name for better human readability
var_name_map <- data.frame(
  Code = c(
    "AG.LND.FRST.K2",
    "BX.KLT.DINV.CD.WD",
    "DT.DOD.DECT.CD",
    "EG.USE.ELEC.KH.PC",
    "NY.GDP.MKTP.CD",
    "NY.GNP.ATLS.CD",
    "NY.GNP.MKTP.PP.CD",
    "SE.SEC.ENRR",
    "SP.ADO.TFRT"
  ),
  Full_Name = c(
    "Forest area (sq. km)",
    "Foreign direct investment, net inflows (BoP, current US$)",
    "External debt stocks, total (DOD, current US$)",
    "Electric power consumption (kWh per capita)",
    "GDP (current US$)",
    "GNI, Atlas method (current US$)",
    "GNI, PPP (current international $)",
    "School enrollment, secondary (% gross)",
    "Adolescent fertility rate (births per 1,000 women ages 15-19)"
  ),
  Short_Name = c(
    "Forest_Area",
    "FDI_Net_Inflows",
    "Debt_Stocks",
    "Electricity_Use",
    "GDP_CurrentUSD",
    "GNI_Atlas",
    "GNI_PPP",
    "Secondary_Enroll",
    "Adolescent_Fertility"
  ),
  stringsAsFactors = FALSE
)

## Joining the short name to our results_df
results_df <- results_df %>%
  left_join(var_name_map, by = c("Dependent_Var" = "Code"))

ggplot(results_df, aes(x = Estimate, y = Short_Name, color = Policy_Var)) +
  geom_point(size = 3) +
  geom_errorbarh(aes(xmin = CI_Lower, xmax = CI_Upper), height = 0.2, linewidth = 0.8) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray40") +
  theme_minimal() +
  labs(
    title = "Estimated Effects of Policies on Development Indicators",
    x = "Coefficient Estimate with 95% CI",
    y = "Development Indicator"
  ) +
  theme(
    legend.position = "bottom",
    panel.grid = element_blank(), # remove grid lines
    panel.background = element_rect(fill = "white"),  # white background
    plot.background = element_rect(fill = "white", color = NA),
    axis.text.y = element_text(size = 7),  
    plot.margin = margin(2, 10, 2, 2) # margin(top, right, bottom, left)  
  )
ggsave("../images/policy_effects_plot.png", width = 7, height = 5, dpi = 300)


#########################################
#### Distibution of the values for the policy variables across years for one country. Only the top 10 countries (based on their average value for that policy variable over time) are shown.

# Reshape the dataframe to long format
long_df <- panel_df %>%
  select(Country.Name, Year, all_of(policy_vars)) %>%
  pivot_longer(cols = all_of(policy_vars), names_to = "Policy_Var", values_to = "Value")

# Average value per country for each policy variable
avg_policy <- long_df %>%
  group_by(Policy_Var, Country.Name) %>%
  summarise(Avg_Value = mean(Value, na.rm = TRUE), .groups = "drop")

# Top 10 countries for each policy variable
top_countries <- avg_policy %>%
  group_by(Policy_Var) %>%
  arrange(desc(Avg_Value)) %>%
  slice_head(n = 10)

# Filter long data for those countries
plot_df <- long_df %>%
  semi_join(top_countries, by = c("Policy_Var", "Country.Name"))

# Horizontal boxplots with facets in one row
ggplot(plot_df, aes(y = reorder(Country.Name, Value, FUN = median), x = Value, fill = Policy_Var)) +
  geom_boxplot(alpha = 0.85) +
  facet_wrap(~ Policy_Var, nrow = 1, scales = "free_x") +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title = "Top 10 Countries by Policy Variable",
    x = "Policy Value", y = "Country"
  ) +
  theme_minimal(base_size = 8) +
  theme(
    axis.text.y = element_text(size = 8),
    axis.text.x = element_text(size = 8),
    panel.grid = element_blank(), # remove grid lines
    panel.background = element_rect(fill = "white"),  # white background
    plot.background = element_rect(fill = "white", color = NA),
    legend.position = "none",
    strip.text = element_text(size = 7, face = "bold")
  )

ggsave("../images/policy_by_country.png", width = 7, height = 4, dpi = 300)

#######################################################
