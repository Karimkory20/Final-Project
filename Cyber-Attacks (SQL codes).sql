use [final project]
--comparing row count
select count(*) as num_of_rows_before_cleaning from [cyber_data_original UTF-8] ;
select count(*) as num_of_rows_after_cleaning from [formatted_cyber] ;


--Checking for Null Values(AttackDate)
select count(*)as null_AttackDate_count_original from [cyber_data_original UTF-8]
where AttackDate is null;
select count(*)as null_AttackDate_count_cleaned from [formatted_cyber]
where AttackDate is null;

--Checking for Null Values(Country)
select count(*)as null_Country_count_original from [cyber_data_original UTF-8]
where Country is null;
select count(*)as null_Country_count_cleaned from [formatted_cyber]
where Country is null;


--Checking for Null Values(spam)
select count(*)as null_spam_count_original from [cyber_data_original UTF-8]
where spam is null;
select count(*)as null_spam_count_cleaned from [formatted_cyber]
where spam is null;

--Checking for Null Values(ramsomwar)
select count(*)as null_ransomware_count_original from [cyber_data_original UTF-8]
where ransomware is null;
select count(*)as null_ransomware_count_cleaned from [formatted_cyber]
where ransomware is null;

--Checking for Null Values(exploit)
select count(*)as null_exploit_count_original from [cyber_data_original UTF-8]
where exploit is null;
select count(*)as null_exploit_count_cleaned from [formatted_cyber]
where exploit is null;


--Checking for Null Values(Malicious_Mail)
select count(*)as null_Malicious_Mail_count_original from [cyber_data_original UTF-8]
where Malicious_Mail is null;
select count(*)as null_Malicious_Mail_count_cleaned from [formatted_cyber]
where Malicious_Mail is null;

--Checking for Null Values(Network_Attack)
select count(*)as null_Network_Attack_count_original from [cyber_data_original UTF-8]
where Network_Attack is null;
select count(*)as null_Network_Attack_count_cleaned from [formatted_cyber]
where Network_Attack is null;


--Checking for Null Values(On_Demand_Scan)
select count(*)as null_On_Demand_Scan_count_original from [cyber_data_original UTF-8]
where On_Demand_Scan is null;
select count(*)as null_On_Demand_Scan_count_cleaned from [formatted_cyber]
where On_Demand_Scan is null;


--Checking for Null Values(Web_Threat)
select count(*)as null_Web_Threat_count_original from [cyber_data_original UTF-8]
where Web_Threat is null;
select count(*)as null_Web_Threat_count_cleaned from [formatted_cyber]
where Web_Threat is null;

--Checking for Null Values(Total_Attack_Percentage)
select count(*)as null_Total_Attack_Percentage_count_original from [cyber_data_original UTF-8]
where Total_Attack_Percentage is null;
select count(*)as null_Total_Attack_Percentage_count_cleaned from [formatted_cyber]
where Total_Attack_Percentage is null;


--Checking for Null Values(Rank_Spam)
select count(*)as null_Rank_Spam_count_original from [cyber_data_original UTF-8]
where Rank_Spam is null;
select count(*)as null_Rank_Spam_count_cleaned from [formatted_cyber]
where Rank_Spam is null;


--Checking for Null Values(Rank_Ransomware)
select count(*)as null_Rank_Ransomware_count_original from [cyber_data_original UTF-8]
where Rank_Ransomware is null;
select count(*)as null_Rank_Ransomware_count_cleaned from [formatted_cyber]
where Rank_Ransomware is null;

--Checking for Null Values(Rank_Local_Infection)
select count(*)as null_Rank_Local_Infection_count_original from [cyber_data_original UTF-8]
where Rank_Local_Infection is null;
select count(*)as null_Rank_Local_Infection_count_cleaned from [formatted_cyber]
where Rank_Local_Infection is null;

--Checking for Null Values(Rank_Exploit)
select count(*)as null_Rank_Exploit_count_original from [cyber_data_original UTF-8]
where Rank_Exploit is null;
select count(*)as null_Rank_Exploit_count_cleaned from [formatted_cyber]
where Rank_Exploit is null;

--Checking for Null Values(Rank_Malicious_Mail)
select count(*)as null_Rank_Malicious_Mail_count_original from [cyber_data_original UTF-8]
where Rank_Malicious_Mail is null;
select count(*)as null_Rank_Malicious_Mail_count_cleaned from [formatted_cyber]
where Rank_Malicious_Mail is null;

--Checking for Null Values(Rank_Network_Attack)
select count(*)as null_Rank_Network_Attack_count_original from [cyber_data_original UTF-8]
where Rank_Network_Attack is null;
select count(*)as null_Rank_Network_Attack_count_cleaned from [formatted_cyber]
where Rank_Network_Attack is null;

--Checking for Null Values(Rank_On_Demand_Scan)
select count(*)as null_Rank_On_Demand_Scan_count_original from [cyber_data_original UTF-8]
where Rank_On_Demand_Scan is null;
select count(*)as null_Rank_On_Demand_Scan_count_cleaned from [formatted_cyber]
where Rank_On_Demand_Scan is null;

--Checking for Null Values(Rank_Web_Threat)
select count(*)as null_Rank_On_Demand_Scan_count_original from [cyber_data_original UTF-8]
where Rank_Web_Threat is null;
select count(*)as null_Rank_Web_Threat_count_cleaned from [formatted_cyber]
where Rank_Web_Threat is null;


--Identifying Duplicate Records(spam)
select spam ,count(*) as num_of_dublicates_original from  [cyber_data_original UTF-8]
group by spam 
having count(*)<1;
select spam ,count(*) as num_of_dublicates_cleaned from  [formatted_cyber]
group by spam 
having count(*)<1;

--Identifying Duplicate Records(Ransomware)
select Ransomware ,count(*) as num_of_original_dublicates from  [cyber_data_original UTF-8]
group by Ransomware 
having count(*)<1;
select Ransomware ,count(*) as num_of_dublicates_cleaned_data from  [formatted_cyber]
group by Ransomware 
having count(*)<1;


--Identifying Duplicate Records(Exploit)
select Exploit ,count(*) as num_of_original_dublicates from  [cyber_data_original UTF-8]
group by Exploit 
having count(*)<1;
select Exploit ,count(*) as num_of_dublicates_cleaned_data from  [formatted_cyber]
group by Exploit 
having count(*)<1;

--Identifying Duplicate Records(Malicious_Mail)
select Malicious_Mail ,count(*) as num_of_original_dublicates from  [cyber_data_original UTF-8]
group by Malicious_Mail 
having count(*)<1;
select Malicious_Mail ,count(*) as num_of_dublicates_cleaned_data from  [formatted_cyber]
group by Malicious_Mail 
having count(*)<1;

--Identifying Duplicate Records(Network_Attack)
select Network_Attack ,count(*) as num_of_original_dublicates from  [cyber_data_original UTF-8]
group by Network_Attack 
having count(*)<1;
select Network_Attack ,count(*) as num_of_dublicates_cleaned_data from  [formatted_cyber]
group by Network_Attack 
having count(*)<1;

--Identifying Duplicate Records(On_Demand_Scan)
select On_Demand_Scan ,count(*) as num_of_original_dublicates from  [cyber_data_original UTF-8]
group by On_Demand_Scan 
having count(*)<1;
select On_Demand_Scan ,count(*) as num_of_dublicates_cleaned_data from  [formatted_cyber]
group by On_Demand_Scan 
having count(*)<1;

--Identifying Duplicate Records(Web_Threat)
select Web_Threat ,count(*) as num_of_original_dublicates from  [cyber_data_original UTF-8]
group by Web_Threat 
having count(*)<1;
select Web_Threat ,count(*) as num_of_dublicates_cleaned_data from  [formatted_cyber]
group by Web_Threat 
having count(*)<1;
